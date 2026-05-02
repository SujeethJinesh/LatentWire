from __future__ import annotations

"""ARC Llama-8B frozen-disagreement source scout.

This is a bounded Mac-local scout for the live stronger-source branch. It does
not claim full source-family generalization: it scores the locally cached
Meta-Llama-3.1-8B-Instruct source only on the already frozen TinyLlama-vs-Qwen
ARC disagreement rows, emits the same 12-byte ARC packet, and compares against
the Qwen-substituted and cached-Tiny packet controls from the parent gate.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_source_private_arc_challenge_fourier_anchor_syndrome_gate as fourier_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_hidden_query_common_basis_gate as hq_gate  # noqa: E402
from scripts import build_source_private_arc_challenge_source_family_cache_falsification as source_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/source_private_arc_challenge_llama8b_disagreement_source_scout_20260502"
)
DEFAULT_TRAIN = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_train.jsonl"
)
DEFAULT_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_TEST = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_test.jsonl"
)
DEFAULT_QWEN_DISAGREEMENT = pathlib.Path(
    "results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu/"
    "qwen_disagreement_predictions.jsonl"
)
DEFAULT_LLAMA_MODEL = pathlib.Path(
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
DEFAULT_SEEDS = (47, 53, 59, 61, 67)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path | str) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: pathlib.Path | str) -> str:
    digest = hashlib.sha256()
    with _resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _load_frozen_rows(
    *,
    validation_path: pathlib.Path,
    test_path: pathlib.Path,
    qwen_disagreement_path: pathlib.Path,
    selection_seed: int,
    validation_limit: int | None,
    test_limit: int | None,
) -> tuple[list[arc_gate.ArcRow], list[arc_gate.ArcRow], list[str], list[str]]:
    validation_full = arc_gate._load_rows(_resolve(validation_path))
    test_full = arc_gate._load_rows(_resolve(test_path))
    validation_ids = hq_gate._load_disagreement_row_ids(
        path=_resolve(qwen_disagreement_path),
        split="validation",
        seed=selection_seed,
        limit=validation_limit,
    )
    test_ids = hq_gate._load_disagreement_row_ids(
        path=_resolve(qwen_disagreement_path),
        split="test",
        seed=selection_seed,
        limit=test_limit,
    )
    return (
        hq_gate._filter_rows_by_ids(validation_full, validation_ids),
        hq_gate._filter_rows_by_ids(test_full, test_ids),
        validation_ids,
        test_ids,
    )


def _source_cache_rows(
    *,
    rows: list[arc_gate.ArcRow],
    predictions: list[int],
    source_family: str,
    source_model: pathlib.Path,
    source_lm_prompt_mode: str,
    source_lm_normalization: str,
) -> list[dict[str, Any]]:
    cache_rows: list[dict[str, Any]] = []
    for row, prediction in zip(rows, predictions, strict=True):
        selected = int(prediction)
        cache_rows.append(
            {
                "row_id": row.row_id,
                "content_id": row.content_id,
                "source_family": source_family,
                "source_model": _display_path(source_model),
                "source_score_mode": "lm_choice_loglikelihood",
                "source_lm_prompt_mode": source_lm_prompt_mode,
                "source_lm_normalization": source_lm_normalization,
                "source_selected_index": selected,
                "source_selected_choice_sha256": arc_gate._sha256_text(row.choices[selected]),
                "source_visible_fields": ["question", "choices"],
                "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            }
        )
    return cache_rows


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _materialize_source_caches(
    *,
    validation_rows: list[arc_gate.ArcRow],
    test_rows: list[arc_gate.ArcRow],
    validation_cache: pathlib.Path,
    test_cache: pathlib.Path,
    source_family: str,
    source_model: pathlib.Path,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_normalization: str,
    source_lm_prompt_mode: str,
    local_files_only: bool,
    force_rematerialize: bool,
) -> dict[str, Any]:
    validation_cache = _resolve(validation_cache)
    test_cache = _resolve(test_cache)
    source_model = _resolve(source_model)
    need_validation = force_rematerialize or not validation_cache.exists()
    need_test = force_rematerialize or not test_cache.exists()
    audit: dict[str, Any] = {
        "source_family": source_family,
        "source_model": _display_path(source_model),
        "materialized_this_run": bool(need_validation or need_test),
        "validation_cache_hit": not need_validation,
        "test_cache_hit": not need_test,
    }
    if not (need_validation or need_test):
        return audit
    score_rows: list[arc_gate.ArcRow] = []
    split_lengths: dict[str, int] = {}
    if need_validation:
        split_lengths["validation"] = len(validation_rows)
        score_rows.extend(validation_rows)
    if need_test:
        split_lengths["test"] = len(test_rows)
        score_rows.extend(test_rows)
    started = time.perf_counter()
    scores, predictions, lm_state = arc_gate._lm_choice_loglikelihood_scores(
        score_rows,
        model_path=str(source_model),
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        local_files_only=local_files_only,
        normalization=source_lm_normalization,
        prompt_mode=source_lm_prompt_mode,
    )
    cursor = 0
    if need_validation:
        count = split_lengths["validation"]
        split_predictions = predictions[cursor : cursor + count]
        _write_jsonl(
            validation_cache,
            _source_cache_rows(
                rows=validation_rows,
                predictions=split_predictions,
                source_family=source_family,
                source_model=source_model,
                source_lm_prompt_mode=source_lm_prompt_mode,
                source_lm_normalization=source_lm_normalization,
            ),
        )
        audit["validation_materialization"] = {
            "rows": count,
            "source_eval_accuracy_before_packet": source_gate._source_accuracy(
                validation_rows, split_predictions
            ),
            "cache": _display_path(validation_cache),
            "cache_sha256": _sha256_file(validation_cache),
        }
        cursor += count
    if need_test:
        count = split_lengths["test"]
        split_predictions = predictions[cursor : cursor + count]
        _write_jsonl(
            test_cache,
            _source_cache_rows(
                rows=test_rows,
                predictions=split_predictions,
                source_family=source_family,
                source_model=source_model,
                source_lm_prompt_mode=source_lm_prompt_mode,
                source_lm_normalization=source_lm_normalization,
            ),
        )
        audit["test_materialization"] = {
            "rows": count,
            "source_eval_accuracy_before_packet": source_gate._source_accuracy(test_rows, split_predictions),
            "cache": _display_path(test_cache),
            "cache_sha256": _sha256_file(test_cache),
        }
    audit["lm_state"] = lm_state
    audit["elapsed_s_total"] = float(time.perf_counter() - started)
    audit["source_score_digest"] = hashlib.sha256(json.dumps(scores, sort_keys=True).encode("utf-8")).hexdigest()
    return audit


def _roll_predictions(rows: list[arc_gate.ArcRow], predictions: list[int]) -> list[int]:
    return [int((prediction + 1) % len(row.choices)) for row, prediction in zip(rows, predictions, strict=True)]


def _random_predictions(rows: list[arc_gate.ArcRow], *, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.integers(0, len(row.choices))) for row in rows]


def _evaluate_prediction_set(
    *,
    split: str,
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    source_features: np.ndarray,
    receiver_features: np.ndarray,
    qwen_disagreement_path: pathlib.Path,
    index_prior: list[float],
    seeds: tuple[int, ...],
    budget_bytes: int,
    code_dim: int,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    return hq_gate._evaluate_features(
        split=split,
        rows=rows,
        source_predictions=source_predictions,
        mapped_features=source_features,
        receiver_features=receiver_features,
        qwen_disagreement_path=_resolve(qwen_disagreement_path),
        index_prior=index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        has_overlap=False,
        variant=variant,
    )


def _pass_split(
    *,
    matched: dict[str, Any],
    roll: dict[str, Any],
    random: dict[str, Any],
    min_gap_over_qwen: float,
    min_gap_over_cached: float,
    min_gap_over_source_control: float,
) -> bool:
    return bool(
        matched["matched_minus_qwen_substituted_mean"] >= min_gap_over_qwen
        and matched["matched_minus_cached_tiny_packet_mean"] >= min_gap_over_cached
        and matched["paired_ci95_low_vs_qwen_substituted_min"] > 0.0
        and matched["paired_ci95_low_vs_cached_tiny_packet_min"] > 0.0
        and matched["matched_accuracy_mean"] >= roll["matched_accuracy_mean"] + min_gap_over_source_control
        and matched["matched_accuracy_mean"] >= random["matched_accuracy_mean"] + min_gap_over_source_control
    )


def _evaluate_split(
    *,
    split: str,
    rows: list[arc_gate.ArcRow],
    train_rows: list[arc_gate.ArcRow],
    source_cache: pathlib.Path,
    qwen_disagreement_path: pathlib.Path,
    seeds: tuple[int, ...],
    budget_bytes: int,
    anchor_count: int,
    spectral_dim: int,
    code_dim: int,
    bootstrap_samples: int,
    min_lift_over_target: float,
    min_gap_over_control: float,
    min_gap_over_text: float,
    min_gap_over_qwen: float,
    min_gap_over_cached: float,
    min_gap_over_source_control: float,
    random_control_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor_texts = arc_gate._choice_pair_texts(train_rows)
    source_features, receiver_features, basis_metadata = fourier_gate._fourier_pair_features_for_variant(
        eval_rows=rows,
        anchor_texts=anchor_texts,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        variant=fourier_gate.MATCHED_VARIANT,
    )
    source_predictions = hq_gate._source_predictions(
        rows,
        hq_gate._read_source_predictions(_resolve(source_cache)),
    )
    index_prior = arc_gate._index_prior(train_rows)
    matched_per_seed, matched_aggregate, matched_rows = _evaluate_prediction_set(
        split=split,
        rows=rows,
        source_predictions=source_predictions,
        source_features=source_features,
        receiver_features=receiver_features,
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        variant="matched_llama8b_source_packet",
    )
    roll_per_seed, roll_aggregate, roll_rows = _evaluate_prediction_set(
        split=split,
        rows=rows,
        source_predictions=_roll_predictions(rows, source_predictions),
        source_features=source_features,
        receiver_features=receiver_features,
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        variant="rolled_source_choice_control",
    )
    random_per_seed, random_aggregate, random_rows = _evaluate_prediction_set(
        split=split,
        rows=rows,
        source_predictions=_random_predictions(rows, seed=random_control_seed),
        source_features=source_features,
        receiver_features=receiver_features,
        qwen_disagreement_path=qwen_disagreement_path,
        index_prior=index_prior,
        seeds=seeds,
        budget_bytes=budget_bytes,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        variant="random_source_choice_control",
    )
    source_accuracy = source_gate._source_accuracy(rows, source_predictions)
    pass_split = _pass_split(
        matched=matched_aggregate,
        roll=roll_aggregate,
        random=random_aggregate,
        min_gap_over_qwen=min_gap_over_qwen,
        min_gap_over_cached=min_gap_over_cached,
        min_gap_over_source_control=min_gap_over_source_control,
    )
    payload = {
        "split": split,
        "rows": len(rows),
        "source_cache": _display_path(source_cache),
        "source_cache_sha256": _sha256_file(source_cache),
        "basis_metadata": basis_metadata,
        "source_accuracy_before_packet": source_accuracy,
        "pass_gate": pass_split,
        "matched_per_seed": matched_per_seed,
        "matched_aggregate": matched_aggregate,
        "rolled_source_choice_per_seed": roll_per_seed,
        "rolled_source_choice_aggregate": roll_aggregate,
        "random_source_choice_per_seed": random_per_seed,
        "random_source_choice_aggregate": random_aggregate,
    }
    return payload, [*matched_rows, *roll_rows, *random_rows]


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    h = payload["headline"]
    lines = [
        "# ARC Llama-8B Disagreement Source Scout",
        "",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source family: `{payload['source_family']}`",
        f"- validation rows: `{h['validation_rows']}`",
        f"- test rows: `{h['test_rows']}`",
        f"- test source accuracy before packet: `{h['test_source_accuracy_before_packet']:.6f}`",
        f"- test matched mean: `{h['test_matched_accuracy_mean']:.6f}`",
        f"- test Qwen-substituted mean: `{h['test_qwen_substituted_accuracy_mean']:.6f}`",
        f"- test cached Tiny mean: `{h['test_cached_tiny_packet_accuracy_mean']:.6f}`",
        f"- test delta vs Qwen-sub: `{h['test_matched_minus_qwen_substituted_mean']:.6f}`",
        f"- test delta vs cached Tiny: `{h['test_matched_minus_cached_tiny_packet_mean']:.6f}`",
        f"- test CI95 low vs Qwen-sub: `{h['test_paired_ci95_low_vs_qwen_substituted_min']:.6f}`",
        f"- test rolled-source control mean: `{h['test_rolled_source_choice_accuracy_mean']:.6f}`",
        f"- test random-source control mean: `{h['test_random_source_choice_accuracy_mean']:.6f}`",
        "",
        "## Lay Explanation",
        "",
        payload["lay_explanation"],
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_split_csv(path: pathlib.Path, payload: dict[str, Any]) -> None:
    fields = [
        "split",
        "variant",
        "pass_gate",
        "rows",
        "source_accuracy_before_packet",
        "matched_accuracy_mean",
        "qwen_substituted_accuracy_mean",
        "cached_tiny_packet_accuracy_mean",
        "matched_minus_qwen_substituted_mean",
        "matched_minus_cached_tiny_packet_mean",
        "paired_ci95_low_vs_qwen_substituted_min",
        "paired_ci95_low_vs_cached_tiny_packet_min",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for split_name, split in payload["splits"].items():
            for variant, key in (
                ("matched", "matched_aggregate"),
                ("rolled_source_choice", "rolled_source_choice_aggregate"),
                ("random_source_choice", "random_source_choice_aggregate"),
            ):
                aggregate = split[key]
                writer.writerow(
                    {
                        "split": split_name,
                        "variant": variant,
                        "pass_gate": split["pass_gate"] if variant == "matched" else "",
                        "rows": split["rows"],
                        "source_accuracy_before_packet": split["source_accuracy_before_packet"],
                        "matched_accuracy_mean": aggregate["matched_accuracy_mean"],
                        "qwen_substituted_accuracy_mean": aggregate["qwen_substituted_accuracy_mean"],
                        "cached_tiny_packet_accuracy_mean": aggregate["cached_tiny_packet_accuracy_mean"],
                        "matched_minus_qwen_substituted_mean": aggregate["matched_minus_qwen_substituted_mean"],
                        "matched_minus_cached_tiny_packet_mean": aggregate[
                            "matched_minus_cached_tiny_packet_mean"
                        ],
                        "paired_ci95_low_vs_qwen_substituted_min": aggregate[
                            "paired_ci95_low_vs_qwen_substituted_min"
                        ],
                        "paired_ci95_low_vs_cached_tiny_packet_min": aggregate[
                            "paired_ci95_low_vs_cached_tiny_packet_min"
                        ],
                    }
                )


def build_scout(
    *,
    output_dir: pathlib.Path = DEFAULT_OUTPUT,
    train_path: pathlib.Path = DEFAULT_TRAIN,
    validation_path: pathlib.Path = DEFAULT_VALIDATION,
    test_path: pathlib.Path = DEFAULT_TEST,
    qwen_disagreement_path: pathlib.Path = DEFAULT_QWEN_DISAGREEMENT,
    source_family: str = "llama3.1_8b_instruct",
    source_model: pathlib.Path = DEFAULT_LLAMA_MODEL,
    source_lm_device: str = "mps",
    source_lm_dtype: str = "float16",
    source_lm_max_length: int = 192,
    source_lm_normalization: str = "mean",
    source_lm_prompt_mode: str = "qa",
    local_files_only: bool = True,
    force_rematerialize: bool = False,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    selection_seed: int = 47,
    validation_limit: int | None = None,
    test_limit: int | None = None,
    budget_bytes: int = 12,
    anchor_count: int = 384,
    spectral_dim: int = 96,
    code_dim: int = 96,
    bootstrap_samples: int = 500,
    min_lift_over_target: float = 0.02,
    min_gap_over_control: float = 0.02,
    min_gap_over_text: float = 0.0,
    min_gap_over_qwen: float = 0.02,
    min_gap_over_cached: float = 0.02,
    min_gap_over_source_control: float = 0.02,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows, test_rows, validation_ids, test_ids = _load_frozen_rows(
        validation_path=validation_path,
        test_path=test_path,
        qwen_disagreement_path=qwen_disagreement_path,
        selection_seed=selection_seed,
        validation_limit=validation_limit,
        test_limit=test_limit,
    )
    validation_cache = output_dir / "llama8b_validation" / "source_prediction_cache.jsonl"
    test_cache = output_dir / "llama8b_test" / "source_prediction_cache.jsonl"
    materialization = _materialize_source_caches(
        validation_rows=validation_rows,
        test_rows=test_rows,
        validation_cache=validation_cache,
        test_cache=test_cache,
        source_family=source_family,
        source_model=source_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_normalization=source_lm_normalization,
        source_lm_prompt_mode=source_lm_prompt_mode,
        local_files_only=local_files_only,
        force_rematerialize=force_rematerialize,
    )
    cache_audit = {
        "validation": source_gate._audit_source_cache(
            rows=validation_rows,
            cache_path=validation_cache,
            label="llama8b_validation_disagreement",
        ),
        "test": source_gate._audit_source_cache(
            rows=test_rows,
            cache_path=test_cache,
            label="llama8b_test_disagreement",
        ),
    }
    train_rows = arc_gate._load_rows(_resolve(train_path))
    validation_payload, validation_predictions = _evaluate_split(
        split="validation",
        rows=validation_rows,
        train_rows=train_rows,
        source_cache=validation_cache,
        qwen_disagreement_path=qwen_disagreement_path,
        seeds=seeds,
        budget_bytes=budget_bytes,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        min_gap_over_qwen=min_gap_over_qwen,
        min_gap_over_cached=min_gap_over_cached,
        min_gap_over_source_control=min_gap_over_source_control,
        random_control_seed=selection_seed + 1009,
    )
    test_payload, test_predictions = _evaluate_split(
        split="test",
        rows=test_rows,
        train_rows=train_rows,
        source_cache=test_cache,
        qwen_disagreement_path=qwen_disagreement_path,
        seeds=seeds,
        budget_bytes=budget_bytes,
        anchor_count=anchor_count,
        spectral_dim=spectral_dim,
        code_dim=code_dim,
        bootstrap_samples=bootstrap_samples,
        min_lift_over_target=min_lift_over_target,
        min_gap_over_control=min_gap_over_control,
        min_gap_over_text=min_gap_over_text,
        min_gap_over_qwen=min_gap_over_qwen,
        min_gap_over_cached=min_gap_over_cached,
        min_gap_over_source_control=min_gap_over_source_control,
        random_control_seed=selection_seed + 2003,
    )
    pass_gate = bool(validation_payload["pass_gate"] and test_payload["pass_gate"])
    headline = {
        "pass_gate": pass_gate,
        "validation_pass": validation_payload["pass_gate"],
        "test_pass": test_payload["pass_gate"],
        "validation_rows": validation_payload["rows"],
        "test_rows": test_payload["rows"],
        "test_source_accuracy_before_packet": test_payload["source_accuracy_before_packet"],
        "test_matched_accuracy_mean": test_payload["matched_aggregate"]["matched_accuracy_mean"],
        "test_qwen_substituted_accuracy_mean": test_payload["matched_aggregate"][
            "qwen_substituted_accuracy_mean"
        ],
        "test_cached_tiny_packet_accuracy_mean": test_payload["matched_aggregate"][
            "cached_tiny_packet_accuracy_mean"
        ],
        "test_matched_minus_qwen_substituted_mean": test_payload["matched_aggregate"][
            "matched_minus_qwen_substituted_mean"
        ],
        "test_matched_minus_cached_tiny_packet_mean": test_payload["matched_aggregate"][
            "matched_minus_cached_tiny_packet_mean"
        ],
        "test_paired_ci95_low_vs_qwen_substituted_min": test_payload["matched_aggregate"][
            "paired_ci95_low_vs_qwen_substituted_min"
        ],
        "test_paired_ci95_low_vs_cached_tiny_packet_min": test_payload["matched_aggregate"][
            "paired_ci95_low_vs_cached_tiny_packet_min"
        ],
        "test_rolled_source_choice_accuracy_mean": test_payload["rolled_source_choice_aggregate"][
            "matched_accuracy_mean"
        ],
        "test_random_source_choice_accuracy_mean": test_payload["random_source_choice_aggregate"][
            "matched_accuracy_mean"
        ],
        "elapsed_s": float(time.perf_counter() - started),
    }
    payload = {
        "gate": "source_private_arc_challenge_llama8b_disagreement_source_scout",
        "date": dt.datetime.now(dt.UTC).date().isoformat(),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pass_gate": pass_gate,
        "pass_rule": (
            "Scout pass requires validation and frozen test matched Llama-8B packets to beat Qwen-substituted "
            "and cached-Tiny packet controls by >= configured margins with positive paired CI95 lower bounds, "
            "and to beat rolled/random source-choice controls by >= configured margins."
        ),
        "source_family": source_family,
        "source_model": _display_path(source_model),
        "method_contract": {
            "scout_not_full_source_family_gate": True,
            "frozen_surface": "TinyLlama-vs-Qwen source-family disagreement rows from seed 47",
            "raw_hidden_transmitted": False,
            "source_text_transmitted": False,
            "source_kv_transmitted": False,
            "packet_format": "12-byte sparse signed ARC Fourier/anchor packet",
            "native_gpu_claims_allowed": False,
        },
        "inputs": {
            "train_path": _display_path(train_path),
            "validation_path": _display_path(validation_path),
            "test_path": _display_path(test_path),
            "qwen_disagreement_path": _display_path(qwen_disagreement_path),
            "qwen_disagreement_sha256": _sha256_file(qwen_disagreement_path),
            "validation_row_ids": validation_ids,
            "test_row_ids": test_ids,
        },
        "materialization": materialization,
        "source_cache_audit": cache_audit,
        "basis": {
            "parent_gate": "results/source_private_arc_challenge_fourier_anchor_syndrome_gate_20260502",
            "anchor_count": anchor_count,
            "spectral_dim": spectral_dim,
            "code_dim": code_dim,
            "budget_bytes": budget_bytes,
        },
        "splits": {
            "validation": validation_payload,
            "test": test_payload,
        },
        "headline": headline,
        "lay_explanation": (
            "This run asks whether a much stronger non-Qwen local source model can choose better ARC answers "
            "on the hard rows where TinyLlama and Qwen disagreed. Only the chosen answer is converted into the "
            "same tiny 12-byte packet; Llama hidden states, text, and KV cache are not transmitted."
        ),
        "interpretation": (
            "A pass would promote Llama-8B to a full source-family gate on all ARC validation/test rows. A "
            "failure would cheaply rule out the only locally cached true non-Qwen stronger source as the next "
            "Mac-local ARC repair, leaving NVIDIA-scale connector training or a new cached source as the live branch."
        ),
    }
    json_path = output_dir / "llama8b_disagreement_source_scout.json"
    md_path = output_dir / "llama8b_disagreement_source_scout.md"
    csv_path = output_dir / "split_summary.csv"
    predictions_path = output_dir / "predictions.jsonl"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    _write_split_csv(csv_path, payload)
    _write_jsonl(predictions_path, [*validation_predictions, *test_predictions])
    manifest = {
        "gate": payload["gate"],
        "pass_gate": pass_gate,
        "headline": headline,
        "files": [
            {"path": _display_path(json_path), "sha256": _sha256_file(json_path)},
            {"path": _display_path(md_path), "sha256": _sha256_file(md_path)},
            {"path": _display_path(csv_path), "sha256": _sha256_file(csv_path)},
            {"path": _display_path(predictions_path), "sha256": _sha256_file(predictions_path)},
            {"path": _display_path(validation_cache), "sha256": _sha256_file(validation_cache)},
            {"path": _display_path(test_cache), "sha256": _sha256_file(test_cache)},
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-path", type=pathlib.Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-path", type=pathlib.Path, default=DEFAULT_TEST)
    parser.add_argument("--qwen-disagreement-path", type=pathlib.Path, default=DEFAULT_QWEN_DISAGREEMENT)
    parser.add_argument("--source-family", default="llama3.1_8b_instruct")
    parser.add_argument("--source-model", type=pathlib.Path, default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--source-lm-device", default="mps")
    parser.add_argument("--source-lm-dtype", default="float16")
    parser.add_argument("--source-lm-max-length", type=int, default=192)
    parser.add_argument("--source-lm-normalization", choices=("mean", "sum"), default="mean")
    parser.add_argument("--source-lm-prompt-mode", choices=("qa", "continuation", "generic_mcq"), default="qa")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--seeds", type=_parse_int_tuple, default="47,53,59,61,67")
    parser.add_argument("--selection-seed", type=int, default=47)
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_scout(
        output_dir=args.output_dir,
        train_path=args.train_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        qwen_disagreement_path=args.qwen_disagreement_path,
        source_family=args.source_family,
        source_model=args.source_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_normalization=args.source_lm_normalization,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        local_files_only=not args.allow_downloads,
        force_rematerialize=args.force_rematerialize,
        seeds=args.seeds,
        selection_seed=args.selection_seed,
        validation_limit=args.validation_limit,
        test_limit=args.test_limit,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "validation_pass": payload["headline"]["validation_pass"],
                "test_pass": payload["headline"]["test_pass"],
                "test_rows": payload["headline"]["test_rows"],
                "test_matched_accuracy_mean": payload["headline"]["test_matched_accuracy_mean"],
                "test_qwen_substituted_accuracy_mean": payload["headline"][
                    "test_qwen_substituted_accuracy_mean"
                ],
                "test_delta_vs_qwen": payload["headline"]["test_matched_minus_qwen_substituted_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
