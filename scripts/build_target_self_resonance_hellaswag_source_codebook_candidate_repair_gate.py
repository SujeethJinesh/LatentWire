from __future__ import annotations

"""Quantized source-codebook candidate repair gate for HellaSwag.

The previous source-residual slot gate showed that a TinyLlama top-1/top-2
score summary can weakly steer a frozen Qwen target-slot baseline, but the
continuous residual slots were too indirect. This gate tests the narrower
receiver-family hypothesis: can a quantized source packet select a receiver
codebook entry that repairs target candidate scores?

The compressed target path receives:

    frozen target-slot candidate scores + source top-1/top-2/margin code

It does not receive source text, source KV, raw source hidden vectors, or raw
source score vectors. The learned object is a receiver-local codebook of
candidate-score repairs keyed by the quantized source code.
"""

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import random
import resource
import statistics
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_target_self_resonance_hellaswag_chunk_encoder_gate as chunk_gate  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_soft_prefix_gate as oracle_gate  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_source_residual_slot_gate as residual_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/target_self_resonance_hellaswag_source_codebook_candidate_repair_gate_20260504_tiny_to_qwen05_train64_validation72_80"
)
DEFAULT_TRAIN_PATH = chunk_gate.DEFAULT_TRAIN_PATH
DEFAULT_EVAL_PATH = chunk_gate.DEFAULT_EVAL_PATH
DEFAULT_TARGET_MODEL = chunk_gate.DEFAULT_TARGET_MODEL
DEFAULT_SOURCE_TRAIN_SCORE_CACHE = residual_gate.DEFAULT_SOURCE_TRAIN_SCORE_CACHE
DEFAULT_SOURCE_EVAL_SCORE_CACHE = residual_gate.DEFAULT_SOURCE_EVAL_SCORE_CACHE

CONDITIONS = (
    "full_prompt",
    "frozen_target_slots",
    "source_codebook_repair",
    "zero_source_codebook",
    "wrong_source_codebook",
    "candidate_roll_source_codebook",
    "target_derived_codebook",
    "random_codebook",
    "source_top1_label_control",
    "source_top2_label_control",
    "source_top1_or_top2_oracle",
    "candidate_derangement",
)

CODE_MODES = (
    "top1",
    "top1_margin",
    "top1_top2_margin",
    "top1_top2_margin_entropy",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return chunk_gate._resolve(path)


def _display(path: pathlib.Path | str) -> str:
    return chunk_gate._display(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    return chunk_gate._sha256_file(path)


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    chunk_gate._write_json(path, payload)


def _write_jsonl(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    chunk_gate._write_jsonl(path, rows)


def _write_csv(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _peak_rss_mib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def _prediction(scores: Sequence[float] | torch.Tensor) -> int:
    return chunk_gate._prediction(scores)


def _kl_to_full(condition_scores: Sequence[float], full_scores: Sequence[float]) -> float:
    return chunk_gate._kl_to_full(condition_scores, full_scores)


def _margin(scores: Sequence[float], answer_index: int) -> float:
    return chunk_gate._margin(scores, answer_index)


def _z_score_vector(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    return centered / (scale if scale > 1e-8 else 1.0)


def _softmax_np(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    shifted = values - float(np.max(values))
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    return exp / denom if denom > 0.0 and math.isfinite(denom) else np.full_like(values, 1.0 / len(values))


def _entropy(scores: Sequence[float]) -> float:
    probs = _softmax_np(scores)
    return -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))) / math.log(len(probs))


def _top2(scores: Sequence[float]) -> tuple[int, int]:
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    return int(order[0]), int(order[1])


def _quantile_bins(values: Sequence[float], *, bin_count: int) -> list[float]:
    if int(bin_count) <= 1:
        return []
    arr = np.asarray(list(values), dtype=np.float64)
    quantiles = [index / int(bin_count) for index in range(1, int(bin_count))]
    return sorted(set(float(value) for value in np.quantile(arr, quantiles)))


def _digitize(value: float, bins: Sequence[float]) -> int:
    return int(np.digitize([float(value)], np.asarray(list(bins), dtype=np.float64))[0])


def _code_parts(
    scores: Sequence[float],
    *,
    margin_bins: Sequence[float],
    entropy_bins: Sequence[float],
) -> tuple[int, int, int, int]:
    values = np.asarray(scores, dtype=np.float64)
    top1, top2 = _top2(values)
    margin = float(values[top1] - values[top2])
    return (
        top1,
        top2,
        _digitize(margin, margin_bins),
        _digitize(_entropy(values), entropy_bins),
    )


def _code_key(parts: tuple[int, int, int, int], *, mode: str) -> tuple[int, ...]:
    top1, top2, margin_bin, entropy_bin = parts
    if mode == "top1":
        return (top1,)
    if mode == "top1_margin":
        return (top1, margin_bin)
    if mode == "top1_top2_margin":
        return (top1, top2, margin_bin)
    if mode == "top1_top2_margin_entropy":
        return (top1, top2, margin_bin, entropy_bin)
    raise ValueError(f"unknown code mode: {mode}")


def _source_top1_scores(source_scores: Sequence[float]) -> list[float]:
    top1, _ = _top2(source_scores)
    return [1.0 if index == top1 else 0.0 for index in range(len(source_scores))]


def _source_top2_scores(source_scores: Sequence[float]) -> list[float]:
    _, top2 = _top2(source_scores)
    return [1.0 if index == top2 else 0.0 for index in range(len(source_scores))]


def _source_pair_oracle_scores(source_scores: Sequence[float], answer_index: int) -> list[float]:
    top1, top2 = _top2(source_scores)
    pred = int(answer_index) if int(answer_index) in {top1, top2} else top1
    return [1.0 if index == pred else 0.0 for index in range(len(source_scores))]


def _fit_codebooks(
    records: Sequence[dict[str, Any]],
    *,
    margin_bins: Sequence[float],
    entropy_bins: Sequence[float],
    mode: str,
    laplace: float,
) -> dict[str, Any]:
    counts_by_code: dict[tuple[int, ...], np.ndarray] = {}
    delta_by_code: dict[tuple[int, ...], np.ndarray] = {}
    code_counts: dict[tuple[int, ...], int] = {}
    global_counts = np.full(4, float(laplace), dtype=np.float64)
    global_delta = np.zeros(4, dtype=np.float64)
    for record in records:
        key = _code_key(
            _code_parts(record["source_scores"], margin_bins=margin_bins, entropy_bins=entropy_bins),
            mode=mode,
        )
        counts_by_code.setdefault(key, np.full(4, float(laplace), dtype=np.float64))
        delta_by_code.setdefault(key, np.zeros(4, dtype=np.float64))
        code_counts[key] = code_counts.get(key, 0) + 1
        answer = int(record["answer_index"])
        counts_by_code[key][answer] += 1.0
        global_counts[answer] += 1.0
        delta = _z_score_vector(record["full_scores"]) - _z_score_vector(record["frozen_target_slots_scores"])
        delta_by_code[key] += delta
        global_delta += delta
    avg_delta_by_code = {
        "|".join(str(part) for part in key): (value / max(1, code_counts[key])).tolist()
        for key, value in delta_by_code.items()
    }
    prior_by_code = {
        "|".join(str(part) for part in key): (value / float(np.sum(value))).tolist()
        for key, value in counts_by_code.items()
    }
    return {
        "mode": mode,
        "laplace": float(laplace),
        "margin_bins": [float(value) for value in margin_bins],
        "entropy_bins": [float(value) for value in entropy_bins],
        "prior_by_code": prior_by_code,
        "delta_by_code": avg_delta_by_code,
        "code_counts": {"|".join(str(part) for part in key): int(value) for key, value in code_counts.items()},
        "global_prior": (global_counts / float(np.sum(global_counts))).tolist(),
        "global_delta": (global_delta / max(1, len(records))).tolist(),
    }


def _lookup_codebook(
    codebook: dict[str, Any],
    scores: Sequence[float] | None,
    *,
    mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if scores is None:
        return (
            np.asarray(codebook["global_prior"], dtype=np.float64),
            np.asarray(codebook["global_delta"], dtype=np.float64),
            "__global__",
        )
    selected_mode = str(mode or codebook["mode"])
    key = _code_key(
        _code_parts(scores, margin_bins=codebook["margin_bins"], entropy_bins=codebook["entropy_bins"]),
        mode=selected_mode,
    )
    parts = list(key)
    prior_by_code = codebook["prior_by_code"]
    delta_by_code = codebook["delta_by_code"]
    while parts:
        encoded = "|".join(str(part) for part in parts)
        if encoded in prior_by_code and encoded in delta_by_code:
            return (
                np.asarray(prior_by_code[encoded], dtype=np.float64),
                np.asarray(delta_by_code[encoded], dtype=np.float64),
                encoded,
            )
        parts.pop()
    return (
        np.asarray(codebook["global_prior"], dtype=np.float64),
        np.asarray(codebook["global_delta"], dtype=np.float64),
        "__global__",
    )


def _repair_scores(
    frozen_scores: Sequence[float],
    source_scores: Sequence[float] | None,
    *,
    codebook: dict[str, Any],
    prior_weight: float,
    delta_weight: float,
) -> tuple[list[float], str]:
    prior, delta, code = _lookup_codebook(codebook, source_scores)
    repaired = (
        _z_score_vector(frozen_scores)
        + float(prior_weight) * np.log(np.clip(prior, 1e-9, 1.0))
        + float(delta_weight) * delta
    )
    return [float(value) for value in repaired], code


def _select_train_rows_from_cache(
    *,
    train_path: pathlib.Path,
    source_train_score_cache: pathlib.Path,
    train_start: int,
    train_rows: int,
) -> tuple[list[arc_gate.ArcRow], list[list[float]], dict[str, Any]]:
    return residual_gate._select_train_rows_from_cache(
        train_path=train_path,
        source_train_score_cache=source_train_score_cache,
        train_start=train_start,
        train_rows=train_rows,
    )


def _select_eval_rows_with_scores(
    *,
    eval_path: pathlib.Path,
    source_eval_score_cache: pathlib.Path,
    eval_start: int,
    eval_rows: int,
) -> tuple[list[arc_gate.ArcRow], list[list[float]], dict[str, Any]]:
    return residual_gate._select_eval_rows_with_scores(
        eval_path=eval_path,
        source_eval_score_cache=source_eval_score_cache,
        eval_start=eval_start,
        eval_rows=eval_rows,
    )


def _records_from_items(
    *,
    items: Sequence[dict[str, Any]],
    source_scores: Sequence[Sequence[float]],
    slots_encoder: torch.nn.Module,
    target_model: Any,
    embed_tokens: Any,
    anchor_ids: torch.Tensor,
    device: str,
    embed_rms: float,
    length_normalize: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for item, scores in zip(items, source_scores, strict=True):
            chunk_prefix = item["chunk_prefix"].to(device=device, dtype=embed_tokens.weight.dtype)
            prefix = chunk_gate._normalize_prefix_rms(slots_encoder(chunk_prefix), embed_rms=embed_rms)
            frozen_scores = [
                float(value)
                for value in chunk_gate._prefix_scores(
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    prefix=prefix,
                    anchor_ids=anchor_ids,
                    choice_ids=item["choice_ids"],
                    device=device,
                    length_normalize=length_normalize,
                ).detach().cpu()
            ]
            row = item["row"]
            records.append(
                {
                    "row": row,
                    "row_id": str(row.row_id),
                    "content_id": str(row.content_id),
                    "answer_index": int(row.answer_index),
                    "answer_label": row.answer_label,
                    "choice_labels": list(row.choice_labels),
                    "full_scores": [float(value) for value in item["full_scores"]],
                    "frozen_target_slots_scores": frozen_scores,
                    "source_scores": [float(value) for value in scores],
                }
            )
    return records


def _score_grid(
    *,
    train_records: Sequence[dict[str, Any]],
    margin_bins: Sequence[float],
    entropy_bins: Sequence[float],
    prior_weights: Sequence[float],
    delta_weights: Sequence[float],
    laplace: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    best_key: tuple[float, float, float, float] | None = None
    best_codebook: dict[str, Any] | None = None
    best_config: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    answers = np.asarray([int(record["answer_index"]) for record in train_records], dtype=np.int64)
    for mode in CODE_MODES:
        codebook = _fit_codebooks(
            train_records,
            margin_bins=margin_bins,
            entropy_bins=entropy_bins,
            mode=mode,
            laplace=laplace,
        )
        for prior_weight in prior_weights:
            for delta_weight in delta_weights:
                predictions: list[int] = []
                agreements: list[int] = []
                kls: list[float] = []
                for record in train_records:
                    repaired, _ = _repair_scores(
                        record["frozen_target_slots_scores"],
                        record["source_scores"],
                        codebook=codebook,
                        prior_weight=prior_weight,
                        delta_weight=delta_weight,
                    )
                    predictions.append(_prediction(repaired))
                    agreements.append(int(_prediction(repaired) == _prediction(record["full_scores"])))
                    kls.append(_kl_to_full(repaired, record["full_scores"]))
                preds = np.asarray(predictions, dtype=np.int64)
                accuracy = float(np.mean(preds == answers))
                agreement = float(statistics.fmean(agreements)) if agreements else 0.0
                mean_kl = float(statistics.fmean(kls)) if kls else 0.0
                config = {
                    "mode": mode,
                    "prior_weight": float(prior_weight),
                    "delta_weight": float(delta_weight),
                    "train_accuracy": accuracy,
                    "train_full_agreement": agreement,
                    "train_mean_kl": mean_kl,
                }
                rows.append(config)
                key = (accuracy, agreement, -mean_kl, -abs(float(prior_weight)))
                if best_key is None or key > best_key:
                    best_key = key
                    best_codebook = codebook
                    best_config = config
    if best_codebook is None or best_config is None:
        raise ValueError("no codebook configuration selected")
    return best_codebook, {"selected": best_config, "grid": rows}


def _add_prediction_rows(
    *,
    prediction_rows: list[dict[str, Any]],
    record: dict[str, Any],
    condition_scores: dict[str, Sequence[float]],
    code_lookup: dict[str, str],
) -> None:
    full_scores = [float(value) for value in record["full_scores"]]
    full_pred = _prediction(full_scores)
    for condition in CONDITIONS:
        raw_scores = [float(value) for value in condition_scores[condition]]
        nonfinite_score = any(not math.isfinite(score) for score in raw_scores)
        scores = [score if math.isfinite(score) else -1.0e9 for score in raw_scores]
        pred = _prediction(scores)
        raw_kl = _kl_to_full(raw_scores, full_scores)
        kl_was_nonfinite = not math.isfinite(raw_kl)
        prediction_rows.append(
            {
                "row_id": record["row_id"],
                "content_id": record["content_id"],
                "condition": condition,
                "answer_index": int(record["answer_index"]),
                "answer_label": record["answer_label"],
                "prediction_index": int(pred),
                "prediction_label": record["choice_labels"][pred],
                "correct": bool(pred == int(record["answer_index"])),
                "full_prompt_prediction_index": int(full_pred),
                "full_prompt_prediction_label": record["choice_labels"][full_pred],
                "agrees_with_full_prompt": bool(pred == full_pred),
                "margin": float(_margin(scores, int(record["answer_index"]))),
                "kl_to_full": float(raw_kl if math.isfinite(raw_kl) else 1.0e9),
                "kl_was_nonfinite": bool(kl_was_nonfinite),
                "nonfinite_score": bool(nonfinite_score),
                "codebook_key": code_lookup.get(condition, ""),
                "scores": scores,
            }
        )


def _condition_metrics(
    prediction_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for row in prediction_rows:
        by_condition.setdefault(str(row["condition"]), []).append(dict(row))
    metrics: dict[str, dict[str, Any]] = {}
    full_rows = by_condition["full_prompt"]
    full_predictions = np.asarray([int(row["prediction_index"]) for row in full_rows], dtype=np.int64)
    answers = np.asarray([int(row["answer_index"]) for row in full_rows], dtype=np.int64)
    for condition in CONDITIONS:
        rows = by_condition[condition]
        predictions = np.asarray([int(row["prediction_index"]) for row in rows], dtype=np.int64)
        correct = predictions == answers if len(predictions) == len(answers) else np.asarray([], dtype=bool)
        full_agreement = predictions == full_predictions if len(predictions) == len(full_predictions) else np.asarray([], dtype=bool)
        kl_values = [float(row["kl_to_full"]) for row in rows if math.isfinite(float(row["kl_to_full"]))]
        metrics[condition] = {
            "n": int(len(rows)),
            "accuracy": float(correct.mean()) if correct.size else 0.0,
            "agreement_with_full_prompt": float(full_agreement.mean()) if full_agreement.size else 0.0,
            "mean_kl_to_full": float(statistics.fmean(kl_values)) if kl_values else 0.0,
            "median_kl_to_full": float(statistics.median(kl_values)) if kl_values else 0.0,
            "nonfinite_kl_count": int(sum(1 for row in rows if bool(row.get("kl_was_nonfinite", False)))),
            "nonfinite_score_row_count": int(sum(1 for row in rows if bool(row.get("nonfinite_score", False)))),
        }
        if condition != "full_prompt" and len(predictions) == len(answers):
            metrics[condition]["paired_vs_full_prompt_accuracy"] = oracle_gate._paired_ci(
                selected=predictions,
                baseline=full_predictions,
                answers=answers,
                seed=seed + len(metrics) * 997,
                samples=bootstrap_samples,
            )
    return metrics


def _predictions_for_condition(prediction_rows: Sequence[dict[str, Any]], condition: str) -> np.ndarray:
    rows = [row for row in prediction_rows if str(row["condition"]) == condition]
    return np.asarray([int(row["prediction_index"]) for row in rows], dtype=np.int64)


def _answers_from_prediction_rows(prediction_rows: Sequence[dict[str, Any]]) -> np.ndarray:
    rows = [row for row in prediction_rows if str(row["condition"]) == "full_prompt"]
    return np.asarray([int(row["answer_index"]) for row in rows], dtype=np.int64)


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    headline = payload["headline"]
    lines = [
        "# Target Self-Resonance HellaSwag Source-Codebook Candidate Repair Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- artifact: `{payload['artifact_dir']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- selected code mode: `{payload['codebook_selection']['selected']['mode']}`",
        f"- selected prior/delta weights: `{payload['codebook_selection']['selected']['prior_weight']}` / "
        f"`{payload['codebook_selection']['selected']['delta_weight']}`",
        f"- source packet raw/framed bytes: `{payload['source_packet_raw_bytes']}` / `{payload['source_packet_framed_bytes']}`",
        "",
        "## Result",
        "",
        f"- source-codebook accuracy: `{metrics['source_codebook_repair']['accuracy']:.6f}`",
        f"- frozen target-slot accuracy: `{metrics['frozen_target_slots']['accuracy']:.6f}`",
        f"- source-top1 label-control accuracy: `{metrics['source_top1_label_control']['accuracy']:.6f}`",
        f"- source-pair oracle accuracy: `{metrics['source_top1_or_top2_oracle']['accuracy']:.6f}`",
        f"- paired CI95 low vs frozen target slots: `{headline['paired_vs_frozen_target_slots']['ci95_low']:.6f}`",
        f"- paired CI95 low vs source-top1: `{headline['paired_vs_source_top1_label_control']['ci95_low']:.6f}`",
        "",
        "## Condition Metrics",
        "",
        "| Condition | Accuracy | Full agreement | Mean KL |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = metrics[condition]
        lines.append(
            f"| `{condition}` | {row['accuracy']:.6f} | {row['agreement_with_full_prompt']:.6f} | "
            f"{row['mean_kl_to_full']:.6f} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], "", "## Next Gate", "", payload["next_exact_gate"], ""])
    _resolve(path).parent.mkdir(parents=True, exist_ok=True)
    _resolve(path).write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    source_train_score_cache: pathlib.Path,
    source_eval_score_cache: pathlib.Path,
    target_model_path: str,
    source_model_family: str,
    train_start: int,
    train_rows: int,
    eval_start: int,
    eval_rows: int,
    prefix_len: int,
    slot_epochs: int,
    lr: float,
    weight_decay: float,
    norm_weight: float,
    seed: int,
    device: str,
    dtype: str,
    max_length: int,
    anchor_text: str,
    continuation_mode: str,
    length_normalize: bool,
    margin_bin_count: int,
    entropy_bin_count: int,
    prior_weights: Sequence[float],
    delta_weights: Sequence[float],
    laplace: float,
    min_delta_vs_frozen: float,
    min_ci_low_vs_frozen: float,
    min_delta_vs_source_top1: float,
    min_ci_low_vs_source_top1: float,
    bootstrap_samples: int,
    local_files_only: bool,
    run_date: str,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = _resolve(train_path)
    eval_path = _resolve(eval_path)
    source_train_score_cache = _resolve(source_train_score_cache)
    source_eval_score_cache = _resolve(source_eval_score_cache)
    selected_train, train_source_scores, train_source_metadata = _select_train_rows_from_cache(
        train_path=train_path,
        source_train_score_cache=source_train_score_cache,
        train_start=train_start,
        train_rows=train_rows,
    )
    selected_eval, eval_source_scores, eval_source_metadata = _select_eval_rows_with_scores(
        eval_path=eval_path,
        source_eval_score_cache=source_eval_score_cache,
        eval_start=eval_start,
        eval_rows=eval_rows,
    )
    content_overlap = sorted({row.content_id for row in selected_train} & {row.content_id for row in selected_eval})
    if content_overlap:
        raise ValueError(f"train/eval content overlap: {content_overlap[:3]}")
    resolved_device = oracle_gate._resolve_device(device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=oracle_gate._torch_dtype(dtype),
    ).to(resolved_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    embed_tokens = model.get_input_embeddings()
    embed_rms = float(embed_tokens.weight.detach().float().pow(2).mean(dim=1).sqrt().median().cpu())
    anchor_ids = oracle_gate._encode_ids(tokenizer, anchor_text, device=resolved_device, add_special_tokens=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_items = residual_gate._build_items(
        rows=selected_train,
        source_scores=train_source_scores,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        target_model=model,
        device=resolved_device,
        prefix_len=prefix_len,
        embed_rms=embed_rms,
        max_length=max_length,
        continuation_mode=continuation_mode,
        length_normalize=length_normalize,
        feature_mode="top2_margin",
    )
    eval_items = residual_gate._build_items(
        rows=selected_eval,
        source_scores=eval_source_scores,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        target_model=model,
        device=resolved_device,
        prefix_len=prefix_len,
        embed_rms=embed_rms,
        max_length=max_length,
        continuation_mode=continuation_mode,
        length_normalize=length_normalize,
        feature_mode="top2_margin",
    )
    train_mean_prefix = torch.stack([item["chunk_prefix"].float() for item in train_items], dim=0).mean(dim=0)
    slots_encoder = chunk_gate.SlotsOnlyEncoder(initial_prefix=train_mean_prefix.to(dtype=embed_tokens.weight.dtype))
    slots_log = chunk_gate._train_encoder(
        encoder=slots_encoder,
        target_model=model,
        embed_tokens=embed_tokens,
        train_items=train_items,
        anchor_ids=anchor_ids,
        device=resolved_device,
        epochs=slot_epochs,
        lr=lr,
        weight_decay=weight_decay,
        norm_weight=norm_weight,
        embed_rms=embed_rms,
        length_normalize=length_normalize,
        seed=seed + 100,
    )
    train_records = _records_from_items(
        items=train_items,
        source_scores=train_source_scores,
        slots_encoder=slots_encoder,
        target_model=model,
        embed_tokens=embed_tokens,
        anchor_ids=anchor_ids,
        device=resolved_device,
        embed_rms=embed_rms,
        length_normalize=length_normalize,
    )
    eval_records = _records_from_items(
        items=eval_items,
        source_scores=eval_source_scores,
        slots_encoder=slots_encoder,
        target_model=model,
        embed_tokens=embed_tokens,
        anchor_ids=anchor_ids,
        device=resolved_device,
        embed_rms=embed_rms,
        length_normalize=length_normalize,
    )

    train_margins = []
    train_entropies = []
    for scores in train_source_scores:
        top1, top2 = _top2(scores)
        train_margins.append(float(scores[top1] - scores[top2]))
        train_entropies.append(_entropy(scores))
    margin_bins = _quantile_bins(train_margins, bin_count=margin_bin_count)
    entropy_bins = _quantile_bins(train_entropies, bin_count=entropy_bin_count)
    codebook, codebook_selection = _score_grid(
        train_records=train_records,
        margin_bins=margin_bins,
        entropy_bins=entropy_bins,
        prior_weights=prior_weights,
        delta_weights=delta_weights,
        laplace=laplace,
    )
    selected_prior_weight = float(codebook_selection["selected"]["prior_weight"])
    selected_delta_weight = float(codebook_selection["selected"]["delta_weight"])

    prediction_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []
    rng = random.Random(seed + 9001)
    random_source_pool = [record["source_scores"] for record in train_records]
    for eval_index, record in enumerate(eval_records):
        source_scores = record["source_scores"]
        frozen_scores = record["frozen_target_slots_scores"]
        method_scores, method_code = _repair_scores(
            frozen_scores,
            source_scores,
            codebook=codebook,
            prior_weight=selected_prior_weight,
            delta_weight=selected_delta_weight,
        )
        zero_scores, zero_code = _repair_scores(
            frozen_scores,
            None,
            codebook=codebook,
            prior_weight=selected_prior_weight,
            delta_weight=selected_delta_weight,
        )
        wrong_scores, wrong_code = _repair_scores(
            frozen_scores,
            eval_records[(eval_index + 1) % len(eval_records)]["source_scores"],
            codebook=codebook,
            prior_weight=selected_prior_weight,
            delta_weight=selected_delta_weight,
        )
        rolled_source = list(np.roll(np.asarray(source_scores, dtype=np.float64), 1))
        rolled_scores, rolled_code = _repair_scores(
            frozen_scores,
            rolled_source,
            codebook=codebook,
            prior_weight=selected_prior_weight,
            delta_weight=selected_delta_weight,
        )
        target_scores, target_code = _repair_scores(
            frozen_scores,
            frozen_scores,
            codebook=codebook,
            prior_weight=selected_prior_weight,
            delta_weight=selected_delta_weight,
        )
        random_scores_source = list(random_source_pool[rng.randrange(len(random_source_pool))])
        random_scores, random_code = _repair_scores(
            frozen_scores,
            random_scores_source,
            codebook=codebook,
            prior_weight=selected_prior_weight,
            delta_weight=selected_delta_weight,
        )
        condition_scores = {
            "full_prompt": record["full_scores"],
            "frozen_target_slots": frozen_scores,
            "source_codebook_repair": method_scores,
            "zero_source_codebook": zero_scores,
            "wrong_source_codebook": wrong_scores,
            "candidate_roll_source_codebook": rolled_scores,
            "target_derived_codebook": target_scores,
            "random_codebook": random_scores,
            "source_top1_label_control": _source_top1_scores(source_scores),
            "source_top2_label_control": _source_top2_scores(source_scores),
            "source_top1_or_top2_oracle": _source_pair_oracle_scores(source_scores, int(record["answer_index"])),
            "candidate_derangement": list(np.roll(np.asarray(method_scores, dtype=np.float64), 1)),
        }
        code_lookup = {
            "source_codebook_repair": method_code,
            "zero_source_codebook": zero_code,
            "wrong_source_codebook": wrong_code,
            "candidate_roll_source_codebook": rolled_code,
            "target_derived_codebook": target_code,
            "random_codebook": random_code,
        }
        _add_prediction_rows(
            prediction_rows=prediction_rows,
            record=record,
            condition_scores=condition_scores,
            code_lookup=code_lookup,
        )
        row_summaries.append(
            {
                "row_id": record["row_id"],
                "content_id": record["content_id"],
                "answer_index": int(record["answer_index"]),
                "full_prompt_prediction": int(_prediction(condition_scores["full_prompt"])),
                "frozen_target_slots_prediction": int(_prediction(condition_scores["frozen_target_slots"])),
                "source_codebook_prediction": int(_prediction(condition_scores["source_codebook_repair"])),
                "source_top1_prediction": int(_prediction(condition_scores["source_top1_label_control"])),
                "source_top2_prediction": int(_prediction(condition_scores["source_top2_label_control"])),
                "codebook_key": method_code,
                "source_codebook_kl_to_full": float(_kl_to_full(method_scores, record["full_scores"])),
                "frozen_target_slots_kl_to_full": float(_kl_to_full(frozen_scores, record["full_scores"])),
            }
        )

    metrics = _condition_metrics(prediction_rows, seed=seed + 4242, bootstrap_samples=bootstrap_samples)
    answers = _answers_from_prediction_rows(prediction_rows)
    method_predictions = _predictions_for_condition(prediction_rows, "source_codebook_repair")
    frozen_predictions = _predictions_for_condition(prediction_rows, "frozen_target_slots")
    source_top1_predictions = _predictions_for_condition(prediction_rows, "source_top1_label_control")
    paired_vs_frozen = oracle_gate._paired_ci(
        selected=method_predictions,
        baseline=frozen_predictions,
        answers=answers,
        seed=seed + 7001,
        samples=bootstrap_samples,
    )
    paired_vs_source_top1 = oracle_gate._paired_ci(
        selected=method_predictions,
        baseline=source_top1_predictions,
        answers=answers,
        seed=seed + 7003,
        samples=bootstrap_samples,
    )
    destructive_controls = (
        "zero_source_codebook",
        "wrong_source_codebook",
        "candidate_roll_source_codebook",
        "target_derived_codebook",
        "random_codebook",
        "source_top1_label_control",
        "source_top2_label_control",
        "candidate_derangement",
    )
    best_destructive_by_accuracy = max(destructive_controls, key=lambda condition: float(metrics[condition]["accuracy"]))
    best_destructive_by_kl = min(
        destructive_controls,
        key=lambda condition: (
            int(metrics[condition]["nonfinite_kl_count"]),
            int(metrics[condition]["nonfinite_score_row_count"]),
            float(metrics[condition]["mean_kl_to_full"]),
        ),
    )
    method = metrics["source_codebook_repair"]
    frozen = metrics["frozen_target_slots"]
    source_top1 = metrics["source_top1_label_control"]
    pass_gate = bool(
        int(method["nonfinite_kl_count"]) == 0
        and int(method["nonfinite_score_row_count"]) == 0
        and float(paired_vs_frozen["mean_delta"]) >= float(min_delta_vs_frozen)
        and float(paired_vs_frozen["ci95_low"]) >= float(min_ci_low_vs_frozen)
        and float(paired_vs_source_top1["mean_delta"]) >= float(min_delta_vs_source_top1)
        and float(paired_vs_source_top1["ci95_low"]) >= float(min_ci_low_vs_source_top1)
        and method["accuracy"] >= metrics[best_destructive_by_accuracy]["accuracy"]
    )
    source_packet_raw_bytes = 1
    source_packet_framed_bytes = source_packet_raw_bytes + 3
    headline = {
        "source_codebook_accuracy": float(method["accuracy"]),
        "source_codebook_agreement": float(method["agreement_with_full_prompt"]),
        "source_codebook_mean_kl": float(method["mean_kl_to_full"]),
        "frozen_target_slots_accuracy": float(frozen["accuracy"]),
        "frozen_target_slots_mean_kl": float(frozen["mean_kl_to_full"]),
        "source_top1_label_accuracy": float(source_top1["accuracy"]),
        "source_top2_label_accuracy": float(metrics["source_top2_label_control"]["accuracy"]),
        "source_top1_or_top2_oracle_accuracy": float(metrics["source_top1_or_top2_oracle"]["accuracy"]),
        "paired_vs_frozen_target_slots": paired_vs_frozen,
        "paired_vs_source_top1_label_control": paired_vs_source_top1,
        "best_destructive_by_accuracy": best_destructive_by_accuracy,
        "best_destructive_accuracy": float(metrics[best_destructive_by_accuracy]["accuracy"]),
        "best_destructive_by_kl": best_destructive_by_kl,
        "best_destructive_mean_kl": float(metrics[best_destructive_by_kl]["mean_kl_to_full"]),
    }
    interpretation = (
        "The quantized source-codebook candidate repair gate passes this held-out slice. It still needs "
        "adjacent slices, seed repeats, and cross-family separation before it can support a paper claim."
        if pass_gate
        else "The quantized source-codebook candidate repair gate does not pass this held-out slice. It "
        "is useful as a direct test of whether a tiny source code can beat target-only and source-copy "
        "shortcuts rather than merely improving KL."
    )
    next_exact_gate = (
        "Run adjacent HellaSwag slices and seeds with the frozen selected codebook, then test a strict cross-family pair."
        if pass_gate
        else "Inspect whether the source top-1/top-2 oracle has enough headroom; if yes, add a calibrated router that "
        "chooses between source top-1, source top-2, and target-local evidence under the same controls."
    )
    payload: dict[str, Any] = {
        "date": run_date,
        "artifact_dir": _display(output_dir),
        "pass_gate": pass_gate,
        "headline": headline,
        "metrics": metrics,
        "slots_log": slots_log,
        "codebook": codebook,
        "codebook_selection": codebook_selection,
        "row_summaries": row_summaries,
        "train_path": _display(train_path),
        "train_sha256": _sha256_file(train_path),
        "eval_path": _display(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "source_train_metadata": train_source_metadata,
        "source_eval_metadata": eval_source_metadata,
        "source_model_family": source_model_family,
        "source_packet_kind": "top1_top2_margin_entropy_code",
        "source_packet_raw_bytes": int(source_packet_raw_bytes),
        "source_packet_framed_bytes": int(source_packet_framed_bytes),
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed": False,
        "source_score_vector_exposed": False,
        "source_score_summary_exposed": True,
        "train_start": int(train_start),
        "train_row_count": int(len(selected_train)),
        "eval_start": int(eval_start),
        "eval_row_count": int(len(selected_eval)),
        "train_eval_content_overlap": int(len(content_overlap)),
        "target_model_path": str(target_model_path),
        "target_device": resolved_device,
        "dtype": dtype,
        "max_length": int(max_length),
        "prefix_len": int(prefix_len),
        "embed_rms_median": float(embed_rms),
        "anchor_text": anchor_text,
        "anchor_token_count": int(anchor_ids.numel()),
        "continuation_mode": continuation_mode,
        "length_normalize": bool(length_normalize),
        "margin_bin_count": int(margin_bin_count),
        "entropy_bin_count": int(entropy_bin_count),
        "pass_criteria": {
            "min_delta_vs_frozen": float(min_delta_vs_frozen),
            "min_ci_low_vs_frozen": float(min_ci_low_vs_frozen),
            "min_delta_vs_source_top1": float(min_delta_vs_source_top1),
            "min_ci_low_vs_source_top1": float(min_ci_low_vs_source_top1),
        },
        "bootstrap_samples": int(bootstrap_samples),
        "runtime_s": float(time.perf_counter() - start_time),
        "peak_rss_mib": float(_peak_rss_mib()),
        "claim_boundary": (
            "This is a Mac-local source-codebook candidate repair probe. Because it operates at candidate "
            "score level, a paper claim must separate it from logit fusion and explicit source-index shortcuts."
        ),
        "interpretation": interpretation,
        "next_exact_gate": next_exact_gate,
    }
    json_path = output_dir / "target_self_resonance_hellaswag_source_codebook_candidate_repair_gate.json"
    md_path = output_dir / "target_self_resonance_hellaswag_source_codebook_candidate_repair_gate.md"
    predictions_path = output_dir / "predictions.jsonl"
    row_summary_path = output_dir / "row_summaries.csv"
    grid_path = output_dir / "codebook_grid.csv"
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    _write_jsonl(predictions_path, prediction_rows)
    _write_csv(row_summary_path, row_summaries)
    _write_csv(grid_path, codebook_selection["grid"])
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            "target_self_resonance_hellaswag_source_codebook_candidate_repair_gate.json": _sha256_file(json_path),
            "target_self_resonance_hellaswag_source_codebook_candidate_repair_gate.md": _sha256_file(md_path),
            "predictions.jsonl": _sha256_file(predictions_path),
            "row_summaries.csv": _sha256_file(row_summary_path),
            "codebook_grid.csv": _sha256_file(grid_path),
        },
        "headline": headline,
        "pass_gate": pass_gate,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return payload


def _parse_float_list(value: str) -> list[float]:
    return [float(part) for part in str(value).split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--source-train-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_TRAIN_SCORE_CACHE)
    parser.add_argument("--source-eval-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_EVAL_SCORE_CACHE)
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--source-model-family", default="TinyLlama")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=64)
    parser.add_argument("--eval-start", type=int, default=72)
    parser.add_argument("--eval-rows", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=8)
    parser.add_argument("--slot-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--norm-weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--anchor-text", default="Continuation:")
    parser.add_argument("--continuation-mode", default="choice", choices=("choice", "label_choice"))
    parser.add_argument("--no-length-normalize", action="store_true")
    parser.add_argument("--margin-bin-count", type=int, default=4)
    parser.add_argument("--entropy-bin-count", type=int, default=4)
    parser.add_argument("--prior-weights", default="0,0.25,0.5,0.75,1,1.5,2,3,4")
    parser.add_argument("--delta-weights", default="0,0.25,0.5,1,1.5,2")
    parser.add_argument("--laplace", type=float, default=0.5)
    parser.add_argument("--min-delta-vs-frozen", type=float, default=0.005)
    parser.add_argument("--min-ci-low-vs-frozen", type=float, default=0.0)
    parser.add_argument("--min-delta-vs-source-top1", type=float, default=0.005)
    parser.add_argument("--min-ci-low-vs-source-top1", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--run-date", default=dt.date.today().isoformat())
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        source_train_score_cache=args.source_train_score_cache,
        source_eval_score_cache=args.source_eval_score_cache,
        target_model_path=args.target_model_path,
        source_model_family=args.source_model_family,
        train_start=args.train_start,
        train_rows=args.train_rows,
        eval_start=args.eval_start,
        eval_rows=args.eval_rows,
        prefix_len=args.prefix_len,
        slot_epochs=args.slot_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        norm_weight=args.norm_weight,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        anchor_text=args.anchor_text,
        continuation_mode=args.continuation_mode,
        length_normalize=not args.no_length_normalize,
        margin_bin_count=args.margin_bin_count,
        entropy_bin_count=args.entropy_bin_count,
        prior_weights=_parse_float_list(args.prior_weights),
        delta_weights=_parse_float_list(args.delta_weights),
        laplace=args.laplace,
        min_delta_vs_frozen=args.min_delta_vs_frozen,
        min_ci_low_vs_frozen=args.min_ci_low_vs_frozen,
        min_delta_vs_source_top1=args.min_delta_vs_source_top1,
        min_ci_low_vs_source_top1=args.min_ci_low_vs_source_top1,
        bootstrap_samples=args.bootstrap_samples,
        local_files_only=not args.allow_download,
        run_date=args.run_date,
    )
    print(json.dumps(payload["headline"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
