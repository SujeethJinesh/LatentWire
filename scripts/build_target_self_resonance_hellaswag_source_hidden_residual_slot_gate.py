from __future__ import annotations

"""Held-out source-hidden residual-slot gate for target self-resonance.

The score-summary residual gate asks whether a tiny source top-2/margin packet
can improve target-native slots. This stricter gate gives the connector a real
source-hidden signal: TinyLlama candidate hidden summaries are mapped into a
small residual over frozen Qwen soft slots, then evaluated against zero-source,
wrong-source, target-derived, random, and label-copy controls.

The compressed target path is:

    frozen target slots + residual(TinyLlama hidden feature) + anchor + candidate

The target model never receives the original HellaSwag context text in the
compressed path.
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

from scripts import build_source_private_hellaswag_hidden_code_packet_scout as hidden_code  # noqa: E402
from scripts import build_source_private_hellaswag_hidden_summary_repair_probe as hidden_summary  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_chunk_encoder_gate as chunk_gate  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_soft_prefix_gate as oracle_gate  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_source_residual_slot_gate as residual_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/target_self_resonance_hellaswag_source_hidden_residual_slot_gate_20260504_"
    "tiny_to_qwen05_train64_validation80_88"
)
DEFAULT_TRAIN_PATH = chunk_gate.DEFAULT_TRAIN_PATH
DEFAULT_EVAL_PATH = chunk_gate.DEFAULT_EVAL_PATH
DEFAULT_TARGET_MODEL = chunk_gate.DEFAULT_TARGET_MODEL
DEFAULT_SOURCE_MODEL = hidden_code.DEFAULT_SOURCE_MODEL
DEFAULT_SOURCE_TRAIN_SCORE_CACHE = residual_gate.DEFAULT_SOURCE_TRAIN_SCORE_CACHE
DEFAULT_SOURCE_EVAL_SCORE_CACHE = residual_gate.DEFAULT_SOURCE_EVAL_SCORE_CACHE
DEFAULT_SOURCE_TRAIN_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260502_tinyllama_train512/"
    "caches/train_sample_seed_2027/source_train_hidden_cache.npz"
)
DEFAULT_SOURCE_EVAL_HIDDEN_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "source_eval_hidden_cache.npz"
)

CONDITIONS = (
    "full_prompt",
    "frozen_target_slots",
    "source_hidden_residual_slots",
    "zero_source_hidden",
    "wrong_source_hidden",
    "candidate_roll_source_hidden",
    "target_score_derived_hidden_template",
    "random_same_norm_residual",
    "source_top1_label_control",
    "source_top1_or_top2_oracle",
    "candidate_derangement",
)

DESTRUCTIVE_CONTROLS = (
    "zero_source_hidden",
    "wrong_source_hidden",
    "candidate_roll_source_hidden",
    "target_score_derived_hidden_template",
    "random_same_norm_residual",
    "source_top1_label_control",
    "candidate_derangement",
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


def _read_json(path: pathlib.Path | str) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _prediction(scores: Sequence[float] | torch.Tensor) -> int:
    return chunk_gate._prediction(scores)


def _kl_to_full(condition_scores: Sequence[float], full_scores: Sequence[float]) -> float:
    return chunk_gate._kl_to_full(condition_scores, full_scores)


def _margin(scores: Sequence[float], answer_index: int) -> float:
    return chunk_gate._margin(scores, answer_index)


def _normalize_prefix_rms(prefix: torch.Tensor, *, embed_rms: float) -> torch.Tensor:
    return chunk_gate._normalize_prefix_rms(prefix, embed_rms=embed_rms)


def _prefix_scores(
    *,
    target_model: Any,
    embed_tokens: Any,
    prefix: torch.Tensor,
    anchor_ids: torch.Tensor,
    choice_ids: Sequence[torch.Tensor],
    device: str,
    length_normalize: bool,
) -> torch.Tensor:
    return chunk_gate._prefix_scores(
        target_model=target_model,
        embed_tokens=embed_tokens,
        prefix=prefix,
        anchor_ids=anchor_ids,
        choice_ids=choice_ids,
        device=device,
        length_normalize=length_normalize,
    )


def _unit_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm < 1e-8 or not math.isfinite(norm):
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32)


def _source_score_features(scores: Sequence[float]) -> np.ndarray:
    return residual_gate._source_packet_features(scores, feature_mode="top2_margin")


def _hidden_source_features(
    *,
    hidden: np.ndarray,
    scores: Sequence[float],
    feature_mode: str,
) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float64)
    if hidden.ndim == 3:
        hidden = hidden[:, 0, :]
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape choices x hidden_dim or choices x layers x hidden_dim")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != hidden.shape[0]:
        raise ValueError("scores must be a 1D vector matching hidden choices")
    order = np.argsort(-values)
    top1 = int(order[0])
    top2 = int(order[1]) if len(order) > 1 else top1
    mean_hidden = np.mean(hidden, axis=0)
    score_features = _source_score_features(values)
    if feature_mode == "mean_top1_delta":
        parts = [
            _unit_vector(mean_hidden),
            _unit_vector(hidden[top1] - mean_hidden),
            score_features,
        ]
    elif feature_mode == "top2_delta":
        parts = [
            _unit_vector(hidden[top1]),
            _unit_vector(hidden[top2]),
            _unit_vector(hidden[top1] - mean_hidden),
            _unit_vector(hidden[top1] - hidden[top2]),
            score_features,
        ]
    else:
        raise ValueError(f"unknown hidden feature mode: {feature_mode}")
    return np.concatenate(parts).astype(np.float32)


def _candidate_roll_hidden_features(
    *,
    hidden: np.ndarray,
    scores: Sequence[float],
    feature_mode: str,
) -> np.ndarray:
    return _hidden_source_features(
        hidden=np.roll(np.asarray(hidden), shift=1, axis=0),
        scores=np.roll(np.asarray(scores, dtype=np.float64), shift=1),
        feature_mode=feature_mode,
    )


def _zero_features(feature_dim: int) -> np.ndarray:
    return np.zeros(int(feature_dim), dtype=np.float32)


def _target_score_derived_features(
    *,
    frozen_scores: Sequence[float],
    feature_dim: int,
    score_feature_dim: int,
) -> np.ndarray:
    features = np.zeros(int(feature_dim), dtype=np.float32)
    score_features = _source_score_features(frozen_scores).astype(np.float32)
    if score_features.shape[0] != int(score_feature_dim):
        raise ValueError("target score feature width changed")
    features[-int(score_feature_dim) :] = score_features
    return features


def _source_top1_scores(source_scores: Sequence[float]) -> list[float]:
    return residual_gate._source_top1_scores(source_scores)


def _source_top1_or_top2_oracle_scores(source_scores: Sequence[float], answer_index: int) -> list[float]:
    values = np.asarray(source_scores, dtype=np.float64)
    order = np.argsort(-values)
    selected = int(answer_index) if int(answer_index) in set(int(index) for index in order[:2]) else int(order[0])
    return [1.0 if index == selected else 0.0 for index in range(len(values))]


def _load_hidden_subset_cache(
    *,
    npz_path: pathlib.Path,
    meta_path: pathlib.Path,
    rows: Sequence[arc_gate.ArcRow],
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not npz_path.exists() or not meta_path.exists():
        return None
    metadata = _read_json(meta_path)
    row_ids = [str(row_id) for row_id in metadata.get("row_ids", [])]
    row_to_index = {row_id: index for index, row_id in enumerate(row_ids)}
    wanted = [str(row.row_id) for row in rows]
    if any(row_id not in row_to_index for row_id in wanted):
        return None
    with np.load(npz_path) as data:
        features = np.asarray(data["features"], dtype=np.float32)
    indices = np.asarray([row_to_index[row_id] for row_id in wanted], dtype=np.int64)
    return features[indices], metadata | {
        "cache_hit": True,
        "cache_npz": _display(npz_path),
        "cache_meta": _display(meta_path),
        "subset_row_count": int(len(indices)),
    }


def _source_hidden_for_rows(
    *,
    rows: Sequence[arc_gate.ArcRow],
    source_hidden_cache: pathlib.Path | None,
    fallback_npz_path: pathlib.Path,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_prompt_mode: str,
    source_lm_layers: tuple[int, ...],
    local_files_only: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if source_hidden_cache is not None:
        cache_path = _resolve(source_hidden_cache)
        cached = _load_hidden_subset_cache(npz_path=cache_path, meta_path=cache_path.with_suffix(".json"), rows=rows)
        if cached is not None:
            return cached
    fallback_npz_path = _resolve(fallback_npz_path)
    return hidden_summary._source_hidden_features(
        list(rows),
        npz_path=fallback_npz_path,
        meta_path=fallback_npz_path.with_suffix(".json"),
        model_path=source_lm_model,
        device=source_lm_device,
        dtype=source_lm_dtype,
        max_length=source_lm_max_length,
        prompt_mode=source_lm_prompt_mode,
        layers=source_lm_layers,
        local_files_only=local_files_only,
    )


def _build_items(
    *,
    rows: Sequence[arc_gate.ArcRow],
    source_scores: Sequence[Sequence[float]],
    source_hidden: np.ndarray,
    tokenizer: Any,
    embed_tokens: Any,
    target_model: Any,
    device: str,
    prefix_len: int,
    embed_rms: float,
    max_length: int,
    continuation_mode: str,
    length_normalize: bool,
    hidden_feature_mode: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row, scores, hidden in zip(rows, source_scores, source_hidden, strict=True):
        item = chunk_gate._encode_row(
            row=row,
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            target_model=target_model,
            device=device,
            prefix_len=prefix_len,
            embed_rms=embed_rms,
            max_length=max_length,
            continuation_mode=continuation_mode,
            length_normalize=length_normalize,
        )
        item["source_scores"] = [float(value) for value in scores]
        item["source_hidden"] = np.asarray(hidden, dtype=np.float32)
        item["source_features"] = _hidden_source_features(
            hidden=hidden,
            scores=scores,
            feature_mode=hidden_feature_mode,
        )
        item["candidate_roll_source_features"] = _candidate_roll_hidden_features(
            hidden=hidden,
            scores=scores,
            feature_mode=hidden_feature_mode,
        )
        items.append(item)
    return items


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
        margins = [float(row["margin"]) for row in rows]
        nonfinite_kl_count = sum(
            1 for row in rows if bool(row.get("kl_was_nonfinite", False)) or not math.isfinite(float(row["kl_to_full"]))
        )
        nonfinite_score_row_count = sum(
            1
            for row in rows
            if bool(row.get("nonfinite_score", False))
            or any(not math.isfinite(float(score)) for score in row.get("scores", ()))
        )
        metrics[condition] = {
            "n": int(len(rows)),
            "accuracy": float(correct.mean()) if correct.size else 0.0,
            "agreement_with_full_prompt": float(full_agreement.mean()) if full_agreement.size else 0.0,
            "mean_kl_to_full": float(statistics.fmean(kl_values)) if kl_values else 0.0,
            "median_kl_to_full": float(statistics.median(kl_values)) if kl_values else 0.0,
            "mean_margin": float(statistics.fmean(margins)) if margins else 0.0,
            "nonfinite_kl_count": int(nonfinite_kl_count),
            "nonfinite_score_row_count": int(nonfinite_score_row_count),
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


def _add_prediction_rows(
    *,
    prediction_rows: list[dict[str, Any]],
    item: dict[str, Any],
    condition_scores: dict[str, Sequence[float]],
) -> None:
    row = item["row"]
    full_scores = [float(value) for value in item["full_scores"]]
    full_pred = _prediction(full_scores)
    for condition in CONDITIONS:
        raw_scores = [float(value) for value in condition_scores[condition]]
        nonfinite_score = any(not math.isfinite(score) for score in raw_scores)
        scores = [score if math.isfinite(score) else -1.0e9 for score in raw_scores]
        pred = _prediction(scores)
        raw_kl = _kl_to_full(raw_scores, full_scores)
        kl_was_nonfinite = not math.isfinite(raw_kl)
        margin = _margin(scores, row.answer_index)
        prediction_rows.append(
            {
                "row_id": row.row_id,
                "content_id": row.content_id,
                "condition": condition,
                "answer_index": int(row.answer_index),
                "answer_label": row.answer_label,
                "prediction_index": int(pred),
                "prediction_label": row.choice_labels[pred],
                "correct": bool(pred == row.answer_index),
                "full_prompt_prediction_index": int(full_pred),
                "full_prompt_prediction_label": row.choice_labels[full_pred],
                "agrees_with_full_prompt": bool(pred == full_pred),
                "margin": float(margin if math.isfinite(margin) else -1.0e9),
                "kl_to_full": float(raw_kl if math.isfinite(raw_kl) else 1.0e9),
                "kl_was_nonfinite": bool(kl_was_nonfinite),
                "nonfinite_score": bool(nonfinite_score),
                "scores": scores,
            }
        )


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    headline = payload["headline"]
    lines = [
        "# Target Self-Resonance HellaSwag Source-Hidden Residual Slot Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- artifact: `{payload['artifact_dir']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source model family: `{payload['source_model_family']}`",
        f"- target model: `{payload['target_model_path']}`",
        f"- train/eval rows: `{payload['train_row_count']}` / `{payload['eval_row_count']}`",
        f"- hidden feature mode: `{payload['hidden_feature_mode']}`",
        f"- source hidden feature fp16 bytes: `{payload['source_feature_raw_bytes_fp16']}`",
        f"- target prefix fp16 bytes: `{payload['prefix_raw_bytes_fp16']}`",
        "",
        "## Result",
        "",
        f"- source-hidden residual accuracy: `{metrics['source_hidden_residual_slots']['accuracy']:.6f}`",
        f"- frozen target-slot accuracy: `{metrics['frozen_target_slots']['accuracy']:.6f}`",
        f"- source-top1 label-control accuracy: `{metrics['source_top1_label_control']['accuracy']:.6f}`",
        f"- source top1/top2 oracle accuracy: `{metrics['source_top1_or_top2_oracle']['accuracy']:.6f}`",
        f"- source-hidden residual mean KL: `{metrics['source_hidden_residual_slots']['mean_kl_to_full']:.6f}`",
        f"- frozen target-slot mean KL: `{metrics['frozen_target_slots']['mean_kl_to_full']:.6f}`",
        f"- best destructive accuracy: `{headline['best_destructive_accuracy']:.6f}` (`{headline['best_destructive_by_accuracy']}`)",
        f"- paired CI95 low vs frozen target slots: `{headline['paired_vs_frozen_target_slots']['ci95_low']:.6f}`",
        "",
        "## Condition Metrics",
        "",
        "| Condition | Accuracy | Full agreement | Mean KL | Nonfinite scores |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = metrics[condition]
        lines.append(
            f"| `{condition}` | {row['accuracy']:.6f} | {row['agreement_with_full_prompt']:.6f} | "
            f"{row['mean_kl_to_full']:.6f} | {row['nonfinite_score_row_count']} |"
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
    source_train_hidden_cache: pathlib.Path | None,
    source_eval_hidden_cache: pathlib.Path | None,
    source_lm_model: str,
    source_lm_device: str,
    source_lm_dtype: str,
    source_lm_max_length: int,
    source_lm_prompt_mode: str,
    source_lm_layers: tuple[int, ...],
    target_model_path: str,
    source_model_family: str,
    train_start: int,
    train_rows: int,
    eval_start: int,
    eval_rows: int,
    prefix_len: int,
    hidden_dim: int,
    slot_epochs: int,
    residual_epochs: int,
    lr: float,
    weight_decay: float,
    norm_weight: float,
    residual_l2_weight: float,
    source_contrastive_weight: float,
    source_contrastive_margin: float,
    initial_residual_gate: float,
    seed: int,
    device: str,
    dtype: str,
    max_length: int,
    anchor_text: str,
    continuation_mode: str,
    length_normalize: bool,
    hidden_feature_mode: str,
    min_delta_vs_frozen: float,
    min_ci_low_vs_frozen: float,
    min_kl_gain_vs_frozen: float,
    max_mean_kl: float,
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
    selected_train, train_source_scores, train_source_metadata = residual_gate._select_train_rows_from_cache(
        train_path=train_path,
        source_train_score_cache=source_train_score_cache,
        train_start=train_start,
        train_rows=train_rows,
    )
    selected_eval, eval_source_scores, eval_source_metadata = residual_gate._select_eval_rows_with_scores(
        eval_path=eval_path,
        source_eval_score_cache=source_eval_score_cache,
        eval_start=eval_start,
        eval_rows=eval_rows,
    )
    content_overlap = sorted({row.content_id for row in selected_train} & {row.content_id for row in selected_eval})
    if content_overlap:
        raise ValueError(f"train/eval content overlap: {content_overlap[:3]}")

    train_hidden, train_hidden_metadata = _source_hidden_for_rows(
        rows=selected_train,
        source_hidden_cache=source_train_hidden_cache,
        fallback_npz_path=output_dir / "source_train_hidden_cache.npz",
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_prompt_mode=source_lm_prompt_mode,
        source_lm_layers=source_lm_layers,
        local_files_only=local_files_only,
    )
    eval_hidden, eval_hidden_metadata = _source_hidden_for_rows(
        rows=selected_eval,
        source_hidden_cache=source_eval_hidden_cache,
        fallback_npz_path=output_dir / "source_eval_hidden_cache.npz",
        source_lm_model=source_lm_model,
        source_lm_device=source_lm_device,
        source_lm_dtype=source_lm_dtype,
        source_lm_max_length=source_lm_max_length,
        source_lm_prompt_mode=source_lm_prompt_mode,
        source_lm_layers=source_lm_layers,
        local_files_only=local_files_only,
    )
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
    embed_dim = int(embed_tokens.embedding_dim)
    embed_rms = float(embed_tokens.weight.detach().float().pow(2).mean(dim=1).sqrt().median().cpu())
    anchor_ids = oracle_gate._encode_ids(tokenizer, anchor_text, device=resolved_device, add_special_tokens=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_items = _build_items(
        rows=selected_train,
        source_scores=train_source_scores,
        source_hidden=train_hidden,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        target_model=model,
        device=resolved_device,
        prefix_len=prefix_len,
        embed_rms=embed_rms,
        max_length=max_length,
        continuation_mode=continuation_mode,
        length_normalize=length_normalize,
        hidden_feature_mode=hidden_feature_mode,
    )
    eval_items = _build_items(
        rows=selected_eval,
        source_scores=eval_source_scores,
        source_hidden=eval_hidden,
        tokenizer=tokenizer,
        embed_tokens=embed_tokens,
        target_model=model,
        device=resolved_device,
        prefix_len=prefix_len,
        embed_rms=embed_rms,
        max_length=max_length,
        continuation_mode=continuation_mode,
        length_normalize=length_normalize,
        hidden_feature_mode=hidden_feature_mode,
    )
    feature_dim = int(np.asarray(train_items[0]["source_features"]).shape[0])
    score_feature_dim = int(_source_score_features(train_items[0]["source_scores"]).shape[0])

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
    residual_encoder = residual_gate.SourceResidualSlotEncoder(
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        prefix_len=prefix_len,
        initial_residual_gate=initial_residual_gate,
    )
    residual_log = residual_gate._train_source_residual_encoder(
        encoder=residual_encoder,
        slots_encoder=slots_encoder,
        target_model=model,
        embed_tokens=embed_tokens,
        train_items=train_items,
        anchor_ids=anchor_ids,
        device=resolved_device,
        epochs=residual_epochs,
        lr=lr,
        weight_decay=weight_decay,
        norm_weight=norm_weight,
        residual_l2_weight=residual_l2_weight,
        source_contrastive_weight=source_contrastive_weight,
        source_contrastive_margin=source_contrastive_margin,
        embed_rms=embed_rms,
        length_normalize=length_normalize,
        seed=seed,
    )

    prediction_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        eval_chunk_prefixes = [
            item["chunk_prefix"].to(resolved_device, dtype=embed_tokens.weight.dtype) for item in eval_items
        ]
        for eval_index, item in enumerate(eval_items):
            row = item["row"]
            chunk_prefix = eval_chunk_prefixes[eval_index]
            base_prefix = _normalize_prefix_rms(slots_encoder(chunk_prefix), embed_rms=embed_rms)
            source_features = torch.as_tensor(item["source_features"], device=resolved_device, dtype=torch.float32)
            source_residual = residual_encoder(source_features).to(dtype=embed_tokens.weight.dtype)
            source_prefix = _normalize_prefix_rms(base_prefix + source_residual, embed_rms=embed_rms)
            zero_features = torch.as_tensor(_zero_features(feature_dim), device=resolved_device, dtype=torch.float32)
            zero_prefix = _normalize_prefix_rms(
                base_prefix + residual_encoder(zero_features).to(dtype=embed_tokens.weight.dtype),
                embed_rms=embed_rms,
            )
            wrong_features = torch.as_tensor(
                eval_items[(eval_index + 1) % len(eval_items)]["source_features"],
                device=resolved_device,
                dtype=torch.float32,
            )
            wrong_prefix = _normalize_prefix_rms(
                base_prefix + residual_encoder(wrong_features).to(dtype=embed_tokens.weight.dtype),
                embed_rms=embed_rms,
            )
            roll_features = torch.as_tensor(
                item["candidate_roll_source_features"],
                device=resolved_device,
                dtype=torch.float32,
            )
            rolled_prefix = _normalize_prefix_rms(
                base_prefix + residual_encoder(roll_features).to(dtype=embed_tokens.weight.dtype),
                embed_rms=embed_rms,
            )
            frozen_scores = [
                float(value)
                for value in _prefix_scores(
                    target_model=model,
                    embed_tokens=embed_tokens,
                    prefix=base_prefix,
                    anchor_ids=anchor_ids,
                    choice_ids=item["choice_ids"],
                    device=resolved_device,
                    length_normalize=length_normalize,
                ).detach().cpu()
            ]
            target_features = torch.as_tensor(
                _target_score_derived_features(
                    frozen_scores=frozen_scores,
                    feature_dim=feature_dim,
                    score_feature_dim=score_feature_dim,
                ),
                device=resolved_device,
                dtype=torch.float32,
            )
            target_derived_prefix = _normalize_prefix_rms(
                base_prefix + residual_encoder(target_features).to(dtype=embed_tokens.weight.dtype),
                embed_rms=embed_rms,
            )
            random_residual = chunk_gate._random_same_norm_prefix(
                reference=source_residual,
                seed=seed * 1009 + eval_start + eval_index,
                device=resolved_device,
            ).to(device=resolved_device, dtype=source_residual.dtype)
            random_prefix = _normalize_prefix_rms(base_prefix + random_residual, embed_rms=embed_rms)
            condition_scores = {
                "full_prompt": [float(value) for value in item["full_scores"]],
                "frozen_target_slots": frozen_scores,
                "source_hidden_residual_slots": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=source_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "zero_source_hidden": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=zero_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "wrong_source_hidden": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=wrong_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "candidate_roll_source_hidden": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=rolled_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "target_score_derived_hidden_template": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=target_derived_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "random_same_norm_residual": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=random_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "source_top1_label_control": _source_top1_scores(item["source_scores"]),
                "source_top1_or_top2_oracle": _source_top1_or_top2_oracle_scores(
                    item["source_scores"],
                    row.answer_index,
                ),
                "candidate_derangement": [],
            }
            condition_scores["candidate_derangement"] = list(np.roll(condition_scores["source_hidden_residual_slots"], 1))
            _add_prediction_rows(prediction_rows=prediction_rows, item=item, condition_scores=condition_scores)
            row_summaries.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "answer_index": int(row.answer_index),
                    "full_prompt_prediction": int(_prediction(condition_scores["full_prompt"])),
                    "frozen_target_slots_prediction": int(_prediction(condition_scores["frozen_target_slots"])),
                    "source_hidden_residual_prediction": int(_prediction(condition_scores["source_hidden_residual_slots"])),
                    "source_top1_prediction": int(_prediction(condition_scores["source_top1_label_control"])),
                    "source_top1_or_top2_oracle_prediction": int(
                        _prediction(condition_scores["source_top1_or_top2_oracle"])
                    ),
                    "source_hidden_residual_kl_to_full": float(
                        _kl_to_full(condition_scores["source_hidden_residual_slots"], condition_scores["full_prompt"])
                    ),
                    "frozen_target_slots_kl_to_full": float(
                        _kl_to_full(condition_scores["frozen_target_slots"], condition_scores["full_prompt"])
                    ),
                    "source_hidden_residual_rms": float(source_residual.float().pow(2).mean().sqrt().cpu()),
                    "source_feature_norm": float(np.linalg.norm(np.asarray(item["source_features"], dtype=np.float64))),
                }
            )

    metrics = _condition_metrics(prediction_rows, seed=seed + 4242, bootstrap_samples=bootstrap_samples)
    answers = _answers_from_prediction_rows(prediction_rows)
    method_predictions = _predictions_for_condition(prediction_rows, "source_hidden_residual_slots")
    frozen_predictions = _predictions_for_condition(prediction_rows, "frozen_target_slots")
    source_label_predictions = _predictions_for_condition(prediction_rows, "source_top1_label_control")
    paired_vs_frozen = oracle_gate._paired_ci(
        selected=method_predictions,
        baseline=frozen_predictions,
        answers=answers,
        seed=seed + 7001,
        samples=bootstrap_samples,
    )
    paired_vs_source_label = oracle_gate._paired_ci(
        selected=method_predictions,
        baseline=source_label_predictions,
        answers=answers,
        seed=seed + 7003,
        samples=bootstrap_samples,
    )
    best_destructive_by_accuracy = max(DESTRUCTIVE_CONTROLS, key=lambda condition: float(metrics[condition]["accuracy"]))
    best_destructive_by_kl = min(
        DESTRUCTIVE_CONTROLS,
        key=lambda condition: (
            int(metrics[condition]["nonfinite_kl_count"]),
            int(metrics[condition]["nonfinite_score_row_count"]),
            float(metrics[condition]["mean_kl_to_full"]),
        ),
    )
    method = metrics["source_hidden_residual_slots"]
    frozen = metrics["frozen_target_slots"]
    kl_gain_vs_frozen = float(frozen["mean_kl_to_full"] - method["mean_kl_to_full"])
    destructive_accuracy = float(metrics[best_destructive_by_accuracy]["accuracy"])
    pass_gate = bool(
        int(method["nonfinite_kl_count"]) == 0
        and int(method["nonfinite_score_row_count"]) == 0
        and float(paired_vs_frozen["mean_delta"]) >= float(min_delta_vs_frozen)
        and float(paired_vs_frozen["ci95_low"]) >= float(min_ci_low_vs_frozen)
        and kl_gain_vs_frozen >= float(min_kl_gain_vs_frozen)
        and method["mean_kl_to_full"] <= float(max_mean_kl)
        and float(method["accuracy"]) > destructive_accuracy
    )
    headline = {
        "source_hidden_residual_accuracy": float(method["accuracy"]),
        "source_hidden_residual_agreement": float(method["agreement_with_full_prompt"]),
        "source_hidden_residual_mean_kl": float(method["mean_kl_to_full"]),
        "source_hidden_residual_nonfinite_kl_count": int(method["nonfinite_kl_count"]),
        "source_hidden_residual_nonfinite_score_row_count": int(method["nonfinite_score_row_count"]),
        "frozen_target_slots_accuracy": float(frozen["accuracy"]),
        "frozen_target_slots_mean_kl": float(frozen["mean_kl_to_full"]),
        "source_top1_label_accuracy": float(metrics["source_top1_label_control"]["accuracy"]),
        "source_top1_or_top2_oracle_accuracy": float(metrics["source_top1_or_top2_oracle"]["accuracy"]),
        "kl_gain_vs_frozen_target_slots": kl_gain_vs_frozen,
        "paired_vs_frozen_target_slots": paired_vs_frozen,
        "paired_vs_source_top1_label_control": paired_vs_source_label,
        "best_destructive_by_accuracy": best_destructive_by_accuracy,
        "best_destructive_accuracy": float(metrics[best_destructive_by_accuracy]["accuracy"]),
        "best_destructive_by_kl": best_destructive_by_kl,
        "best_destructive_mean_kl": float(metrics[best_destructive_by_kl]["mean_kl_to_full"]),
    }
    interpretation = (
        "The source-hidden residual-slot gate passes this small held-out slice. This promotes adjacent "
        "slices, seed repeats, and a byte-reduced hidden codec."
        if pass_gate
        else "The source-hidden residual-slot gate does not pass this held-out slice. This weakens the "
        "direct hidden-to-target-soft-slot branch unless row-level failures show a clear fix."
    )
    next_exact_gate = (
        "Repeat adjacent validation slices and seeds, then quantize/project the hidden feature before claiming a systems path."
        if pass_gate
        else "Analyze row-level failures, then decide between a smaller SAE/PCA hidden code, an oracle-prefix distillation target, "
        "or a consistency-refined target-native slot interface."
    )
    source_feature_raw_bytes_fp16 = int(feature_dim) * 2
    payload: dict[str, Any] = {
        "date": run_date,
        "artifact_dir": _display(output_dir),
        "pass_gate": pass_gate,
        "headline": headline,
        "metrics": metrics,
        "slots_log": slots_log,
        "residual_log": residual_log,
        "row_summaries": row_summaries,
        "train_path": _display(train_path),
        "train_sha256": _sha256_file(train_path),
        "eval_path": _display(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "source_train_score_metadata": train_source_metadata,
        "source_eval_score_metadata": eval_source_metadata,
        "source_train_hidden_metadata": train_hidden_metadata,
        "source_eval_hidden_metadata": eval_hidden_metadata,
        "source_model_family": source_model_family,
        "source_lm_model": str(source_lm_model),
        "source_lm_device": source_lm_device,
        "source_lm_dtype": source_lm_dtype,
        "source_lm_max_length": int(source_lm_max_length),
        "source_lm_prompt_mode": source_lm_prompt_mode,
        "source_lm_layers": list(source_lm_layers),
        "hidden_feature_mode": hidden_feature_mode,
        "source_feature_dim": int(feature_dim),
        "source_score_feature_dim": int(score_feature_dim),
        "source_feature_raw_bytes_fp16": int(source_feature_raw_bytes_fp16),
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed_to_bridge": True,
        "source_score_summary_exposed_to_bridge": True,
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
        "hidden_dim": int(hidden_dim),
        "embed_dim": int(embed_dim),
        "prefix_raw_bytes_fp16": int(prefix_len) * int(embed_dim) * 2,
        "residual_encoder_parameter_count": int(sum(param.numel() for param in residual_encoder.parameters())),
        "initial_residual_gate": float(initial_residual_gate),
        "slots_parameter_count": int(sum(param.numel() for param in slots_encoder.parameters())),
        "embed_rms_median": float(embed_rms),
        "anchor_text": anchor_text,
        "anchor_token_count": int(anchor_ids.numel()),
        "continuation_mode": continuation_mode,
        "length_normalize": bool(length_normalize),
        "pass_criteria": {
            "min_delta_vs_frozen": float(min_delta_vs_frozen),
            "min_ci_low_vs_frozen": float(min_ci_low_vs_frozen),
            "min_kl_gain_vs_frozen": float(min_kl_gain_vs_frozen),
            "max_mean_kl": float(max_mean_kl),
        },
        "bootstrap_samples": int(bootstrap_samples),
        "runtime_s": float(time.perf_counter() - start_time),
        "peak_rss_mib": float(_peak_rss_mib()),
        "claim_boundary": (
            "This is a Mac-local source-hidden residual-slot probe. It is not yet a systems win: the "
            "current hidden feature is larger than the target prefix and must be projected/quantized."
        ),
        "interpretation": interpretation,
        "next_exact_gate": next_exact_gate,
    }
    json_path = output_dir / "target_self_resonance_hellaswag_source_hidden_residual_slot_gate.json"
    md_path = output_dir / "target_self_resonance_hellaswag_source_hidden_residual_slot_gate.md"
    predictions_path = output_dir / "predictions.jsonl"
    row_summary_path = output_dir / "row_summaries.csv"
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    _write_jsonl(predictions_path, prediction_rows)
    _write_csv(row_summary_path, row_summaries)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            "target_self_resonance_hellaswag_source_hidden_residual_slot_gate.json": _sha256_file(json_path),
            "target_self_resonance_hellaswag_source_hidden_residual_slot_gate.md": _sha256_file(md_path),
            "predictions.jsonl": _sha256_file(predictions_path),
            "row_summaries.csv": _sha256_file(row_summary_path),
        },
        "headline": headline,
        "pass_gate": pass_gate,
    }
    for cache_name in ("source_train_hidden_cache.npz", "source_train_hidden_cache.json", "source_eval_hidden_cache.npz", "source_eval_hidden_cache.json"):
        cache_path = output_dir / cache_name
        if cache_path.exists():
            manifest["files"][cache_name] = _sha256_file(cache_path)
    _write_json(output_dir / "manifest.json", manifest)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-path", type=pathlib.Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--source-train-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_TRAIN_SCORE_CACHE)
    parser.add_argument("--source-eval-score-cache", type=pathlib.Path, default=DEFAULT_SOURCE_EVAL_SCORE_CACHE)
    parser.add_argument("--source-train-hidden-cache", type=pathlib.Path, default=DEFAULT_SOURCE_TRAIN_HIDDEN_CACHE)
    parser.add_argument("--source-eval-hidden-cache", type=pathlib.Path, default=DEFAULT_SOURCE_EVAL_HIDDEN_CACHE)
    parser.add_argument("--source-lm-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-lm-device", default="auto")
    parser.add_argument("--source-lm-dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--source-lm-max-length", type=int, default=256)
    parser.add_argument("--source-lm-prompt-mode", default="continuation")
    parser.add_argument("--source-lm-layers", type=_parse_int_tuple, default=(-1,))
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--source-model-family", default="TinyLlama")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=64)
    parser.add_argument("--eval-start", type=int, default=80)
    parser.add_argument("--eval-rows", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--slot-epochs", type=int, default=3)
    parser.add_argument("--residual-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--norm-weight", type=float, default=0.001)
    parser.add_argument("--residual-l2-weight", type=float, default=0.001)
    parser.add_argument("--source-contrastive-weight", type=float, default=0.2)
    parser.add_argument("--source-contrastive-margin", type=float, default=0.05)
    parser.add_argument("--initial-residual-gate", type=float, default=-5.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--anchor-text", default="Continuation:")
    parser.add_argument("--continuation-mode", choices=("choice", "label_and_choice", "label"), default="choice")
    parser.add_argument("--length-normalize", choices=("true", "false"), default="true")
    parser.add_argument("--hidden-feature-mode", choices=("top2_delta", "mean_top1_delta"), default="top2_delta")
    parser.add_argument("--min-delta-vs-frozen", type=float, default=0.001)
    parser.add_argument("--min-ci-low-vs-frozen", type=float, default=0.0)
    parser.add_argument("--min-kl-gain-vs-frozen", type=float, default=0.0)
    parser.add_argument("--max-mean-kl", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--run-date", default=str(dt.date.today()))
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        source_train_score_cache=args.source_train_score_cache,
        source_eval_score_cache=args.source_eval_score_cache,
        source_train_hidden_cache=args.source_train_hidden_cache,
        source_eval_hidden_cache=args.source_eval_hidden_cache,
        source_lm_model=args.source_lm_model,
        source_lm_device=args.source_lm_device,
        source_lm_dtype=args.source_lm_dtype,
        source_lm_max_length=args.source_lm_max_length,
        source_lm_prompt_mode=args.source_lm_prompt_mode,
        source_lm_layers=args.source_lm_layers,
        target_model_path=args.target_model_path,
        source_model_family=args.source_model_family,
        train_start=args.train_start,
        train_rows=args.train_rows,
        eval_start=args.eval_start,
        eval_rows=args.eval_rows,
        prefix_len=args.prefix_len,
        hidden_dim=args.hidden_dim,
        slot_epochs=args.slot_epochs,
        residual_epochs=args.residual_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        norm_weight=args.norm_weight,
        residual_l2_weight=args.residual_l2_weight,
        source_contrastive_weight=args.source_contrastive_weight,
        source_contrastive_margin=args.source_contrastive_margin,
        initial_residual_gate=args.initial_residual_gate,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        anchor_text=args.anchor_text,
        continuation_mode=args.continuation_mode,
        length_normalize=args.length_normalize == "true",
        hidden_feature_mode=args.hidden_feature_mode,
        min_delta_vs_frozen=args.min_delta_vs_frozen,
        min_ci_low_vs_frozen=args.min_ci_low_vs_frozen,
        min_kl_gain_vs_frozen=args.min_kl_gain_vs_frozen,
        max_mean_kl=args.max_mean_kl,
        bootstrap_samples=args.bootstrap_samples,
        local_files_only=args.local_files_only == "true",
        run_date=args.run_date,
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "headline": payload["headline"],
                "artifact_dir": payload["artifact_dir"],
                "runtime_s": payload["runtime_s"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
