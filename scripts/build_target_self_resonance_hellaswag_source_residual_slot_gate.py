from __future__ import annotations

"""Held-out source-conditioned residual-slot gate for target self-resonance.

Target-only soft-slot encoders failed to separate from cache-like controls. This
gate asks the next narrower question: can a small source-derived code add a
useful residual to a frozen target-slot baseline?

The compressed target path receives:

    frozen target slots + learned residual(source top-2/margin code) + anchor + candidate

It never receives the original context text. The default source code is a
TinyLlama score-summary packet and the default target is Qwen2.5-0.5B, so this
is a Mac-local cross-family residual-interface probe.
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
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/target_self_resonance_hellaswag_source_residual_slot_gate_20260504_tiny_to_qwen05_train64_validation72_80"
)
DEFAULT_TRAIN_PATH = chunk_gate.DEFAULT_TRAIN_PATH
DEFAULT_EVAL_PATH = chunk_gate.DEFAULT_EVAL_PATH
DEFAULT_TARGET_MODEL = chunk_gate.DEFAULT_TARGET_MODEL
DEFAULT_SOURCE_TRAIN_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_train_sample_stress_20260502_tinyllama_train512/"
    "caches/train_sample_seed_2027/source_train_score_cache.json"
)
DEFAULT_SOURCE_EVAL_SCORE_CACHE = pathlib.Path(
    "results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/"
    "source_eval_score_cache.json"
)

CONDITIONS = (
    "full_prompt",
    "frozen_target_slots",
    "source_residual_slots",
    "zero_source_residual",
    "wrong_source_residual",
    "candidate_roll_source_residual",
    "target_derived_residual",
    "random_same_norm_residual",
    "source_top1_label_control",
    "candidate_derangement",
)


class SourceResidualSlotEncoder(torch.nn.Module):
    """Map a compact source code to a small residual over frozen target slots."""

    def __init__(
        self,
        *,
        feature_dim: int,
        embed_dim: int,
        hidden_dim: int,
        prefix_len: int,
        initial_residual_gate: float = -5.0,
    ) -> None:
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.embed_dim = int(embed_dim)
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(int(feature_dim)),
            torch.nn.Linear(int(feature_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(hidden_dim), int(prefix_len) * int(embed_dim), bias=False),
        )
        self.residual_gate = torch.nn.Parameter(torch.tensor(float(initial_residual_gate)))

    def forward(self, source_features: torch.Tensor) -> torch.Tensor:
        if source_features.dim() != 1:
            raise ValueError("source_features must be a 1D tensor")
        residual = self.net(source_features.float()).view(self.prefix_len, self.embed_dim)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.sigmoid(self.residual_gate) * torch.tanh(residual)


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


def _softmax_np(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    denom = float(np.sum(exp))
    return exp / denom if denom > 0.0 and math.isfinite(denom) else np.full_like(scores, 1.0 / len(scores))


def _source_packet_features(scores: Sequence[float], *, feature_mode: str = "top2_margin") -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("source scores must be a 1D vector with at least two choices")
    if feature_mode == "score_z":
        centered = values - float(np.mean(values))
        scale = float(np.std(centered))
        return (centered / (scale if scale > 1e-8 else 1.0)).astype(np.float32)
    if feature_mode != "top2_margin":
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    order = np.argsort(-values)
    top1 = int(order[0])
    top2 = int(order[1])
    probs = _softmax_np(values)
    one_hot_top1 = np.eye(values.size, dtype=np.float32)[top1]
    one_hot_top2 = np.eye(values.size, dtype=np.float32)[top2]
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    margin = float(values[top1] - values[top2]) / (scale if scale > 1e-8 else 1.0)
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))) / math.log(values.size)
    return np.concatenate(
        [
            one_hot_top1,
            one_hot_top2,
            np.asarray([math.tanh(margin), entropy], dtype=np.float32),
        ]
    ).astype(np.float32)


def _zero_features(feature_dim: int) -> np.ndarray:
    return np.zeros(int(feature_dim), dtype=np.float32)


def _source_top1_scores(source_scores: Sequence[float]) -> list[float]:
    top = int(np.argmax(np.asarray(source_scores, dtype=np.float64)))
    return [1.0 if index == top else 0.0 for index in range(len(source_scores))]


def _load_score_cache(path: pathlib.Path | str) -> dict[str, tuple[int, list[float]]]:
    cache = _read_json(path)
    return {
        str(row_id): (int(prediction), [float(value) for value in scores])
        for row_id, prediction, scores in zip(
            cache["row_ids"],
            cache["source_predictions"],
            cache["source_scores"],
            strict=True,
        )
    }


def _select_train_rows_from_cache(
    *,
    train_path: pathlib.Path,
    source_train_score_cache: pathlib.Path,
    train_start: int,
    train_rows: int,
) -> tuple[list[arc_gate.ArcRow], list[list[float]], dict[str, Any]]:
    all_train_rows = arc_gate._load_rows(train_path)
    by_row_id = {str(row.row_id): row for row in all_train_rows}
    cache = _read_json(source_train_score_cache)
    selected_ids = [str(row_id) for row_id in cache["row_ids"][int(train_start) : int(train_start) + int(train_rows)]]
    if len(selected_ids) < int(train_rows):
        raise ValueError("source train score cache does not contain enough rows")
    missing = [row_id for row_id in selected_ids if row_id not in by_row_id]
    if missing:
        raise ValueError(f"missing official train rows for source ids: {missing[:3]}")
    score_map = _load_score_cache(source_train_score_cache)
    return (
        [by_row_id[row_id] for row_id in selected_ids],
        [score_map[row_id][1] for row_id in selected_ids],
        {
            "source_train_score_cache": _display(source_train_score_cache),
            "source_train_score_cache_sha256": _sha256_file(source_train_score_cache),
            "source_train_score_cache_rows": int(cache["row_count"]),
        },
    )


def _select_eval_rows_with_scores(
    *,
    eval_path: pathlib.Path,
    source_eval_score_cache: pathlib.Path,
    eval_start: int,
    eval_rows: int,
) -> tuple[list[arc_gate.ArcRow], list[list[float]], dict[str, Any]]:
    selected_eval = chunk_gate._row_slice(arc_gate._load_rows(eval_path), start=eval_start, limit=eval_rows)
    score_map = _load_score_cache(source_eval_score_cache)
    missing = [str(row.row_id) for row in selected_eval if str(row.row_id) not in score_map]
    if missing:
        raise ValueError(f"missing source eval scores for {len(missing)} rows")
    cache = _read_json(source_eval_score_cache)
    return (
        selected_eval,
        [score_map[str(row.row_id)][1] for row in selected_eval],
        {
            "source_eval_score_cache": _display(source_eval_score_cache),
            "source_eval_score_cache_sha256": _sha256_file(source_eval_score_cache),
            "source_eval_score_cache_rows": int(cache["row_count"]),
        },
    )


def _build_items(
    *,
    rows: Sequence[arc_gate.ArcRow],
    source_scores: Sequence[Sequence[float]],
    tokenizer: Any,
    embed_tokens: Any,
    target_model: Any,
    device: str,
    prefix_len: int,
    embed_rms: float,
    max_length: int,
    continuation_mode: str,
    length_normalize: bool,
    feature_mode: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row, scores in zip(rows, source_scores, strict=True):
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
        item["source_features"] = _source_packet_features(scores, feature_mode=feature_mode)
        item["candidate_roll_source_features"] = _source_packet_features(np.roll(scores, 1), feature_mode=feature_mode)
        items.append(item)
    return items


def _contrastive_kl_penalty(pos_kl: torch.Tensor, neg_kl: torch.Tensor, *, margin: float) -> torch.Tensor:
    return torch.nn.functional.softplus(torch.as_tensor(float(margin), device=pos_kl.device) + pos_kl - neg_kl)


def _train_source_residual_encoder(
    *,
    encoder: SourceResidualSlotEncoder,
    slots_encoder: torch.nn.Module,
    target_model: Any,
    embed_tokens: Any,
    train_items: Sequence[dict[str, Any]],
    anchor_ids: torch.Tensor,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    norm_weight: float,
    residual_l2_weight: float,
    source_contrastive_weight: float,
    source_contrastive_margin: float,
    embed_rms: float,
    length_normalize: bool,
    seed: int,
) -> dict[str, Any]:
    encoder.to(device)
    slots_encoder.to(device)
    slots_encoder.eval()
    for param in slots_encoder.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = random.Random(int(seed))
    losses: list[float] = []
    kls: list[float] = []
    negative_kls: list[float] = []
    residual_rms_values: list[float] = []
    for epoch in range(int(epochs)):
        indices = list(range(len(train_items)))
        rng.shuffle(indices)
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_negative_kl = 0.0
        epoch_residual_rms = 0.0
        for idx in indices:
            item = train_items[idx]
            optimizer.zero_grad(set_to_none=True)
            chunk_prefix = item["chunk_prefix"].to(device=device, dtype=embed_tokens.weight.dtype)
            base_prefix = _normalize_prefix_rms(slots_encoder(chunk_prefix), embed_rms=embed_rms)
            source_features = torch.as_tensor(item["source_features"], device=device, dtype=torch.float32)
            residual = encoder(source_features).to(dtype=embed_tokens.weight.dtype)
            prefix = _normalize_prefix_rms(base_prefix + residual, embed_rms=embed_rms)
            scores = _prefix_scores(
                target_model=target_model,
                embed_tokens=embed_tokens,
                prefix=prefix,
                anchor_ids=anchor_ids,
                choice_ids=item["choice_ids"],
                device=device,
                length_normalize=length_normalize,
            )
            full_probs = item["full_probs"].to(device)
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(scores.float(), dim=-1),
                full_probs,
                reduction="sum",
            )
            negative_kl = torch.zeros((), dtype=kl.dtype, device=kl.device)
            contrastive_loss = torch.zeros((), dtype=kl.dtype, device=kl.device)
            if float(source_contrastive_weight) > 0.0 and len(train_items) > 1:
                wrong_idx = rng.randrange(len(train_items) - 1)
                if wrong_idx >= idx:
                    wrong_idx += 1
                wrong_features = torch.as_tensor(train_items[wrong_idx]["source_features"], device=device, dtype=torch.float32)
                wrong_prefix = _normalize_prefix_rms(
                    base_prefix + encoder(wrong_features).to(dtype=embed_tokens.weight.dtype),
                    embed_rms=embed_rms,
                )
                wrong_scores = _prefix_scores(
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    prefix=wrong_prefix,
                    anchor_ids=anchor_ids,
                    choice_ids=item["choice_ids"],
                    device=device,
                    length_normalize=length_normalize,
                )
                negative_kl = torch.nn.functional.kl_div(
                    torch.log_softmax(wrong_scores.float(), dim=-1),
                    full_probs,
                    reduction="sum",
                )
                contrastive_loss = _contrastive_kl_penalty(kl, negative_kl, margin=source_contrastive_margin)
            residual_l2 = residual.float().pow(2).mean()
            loss = (
                kl
                + float(norm_weight) * chunk_gate._prefix_rms_loss(prefix, embed_rms=embed_rms)
                + float(residual_l2_weight) * residual_l2
                + float(source_contrastive_weight) * contrastive_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite source-residual loss at epoch={epoch} row_index={idx}")
            epoch_loss += float(loss.detach().cpu())
            epoch_kl += float(kl.detach().cpu())
            epoch_negative_kl += float(negative_kl.detach().cpu())
            epoch_residual_rms += float(residual.float().pow(2).mean().sqrt().detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
        denom = max(1, len(train_items))
        losses.append(epoch_loss / denom)
        kls.append(epoch_kl / denom)
        negative_kls.append(epoch_negative_kl / denom)
        residual_rms_values.append(epoch_residual_rms / denom)
    encoder.eval()
    return {
        "loss_initial": float(losses[0]) if losses else 0.0,
        "loss_final": float(losses[-1]) if losses else 0.0,
        "kl_initial": float(kls[0]) if kls else 0.0,
        "kl_final": float(kls[-1]) if kls else 0.0,
        "negative_kl_initial": float(negative_kls[0]) if negative_kls else 0.0,
        "negative_kl_final": float(negative_kls[-1]) if negative_kls else 0.0,
        "residual_rms_initial": float(residual_rms_values[0]) if residual_rms_values else 0.0,
        "residual_rms_final": float(residual_rms_values[-1]) if residual_rms_values else 0.0,
        "residual_gate": float(torch.sigmoid(encoder.residual_gate.detach()).cpu()),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "norm_weight": float(norm_weight),
        "residual_l2_weight": float(residual_l2_weight),
        "source_contrastive_weight": float(source_contrastive_weight),
        "source_contrastive_margin": float(source_contrastive_margin),
    }


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
        "# Target Self-Resonance HellaSwag Source-Residual Slot Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- artifact: `{payload['artifact_dir']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source model family: `{payload['source_model_family']}`",
        f"- target model: `{payload['target_model_path']}`",
        f"- train/eval rows: `{payload['train_row_count']}` / `{payload['eval_row_count']}`",
        f"- source feature mode: `{payload['source_feature_mode']}`",
        f"- source packet raw/framed bytes: `{payload['source_packet_raw_bytes']}` / `{payload['source_packet_framed_bytes']}`",
        "",
        "## Result",
        "",
        f"- source-residual accuracy: `{metrics['source_residual_slots']['accuracy']:.6f}`",
        f"- frozen target-slot accuracy: `{metrics['frozen_target_slots']['accuracy']:.6f}`",
        f"- source-top1 label-control accuracy: `{metrics['source_top1_label_control']['accuracy']:.6f}`",
        f"- source-residual mean KL: `{metrics['source_residual_slots']['mean_kl_to_full']:.6f}`",
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
    source_feature_mode: str,
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
    embed_dim = int(embed_tokens.embedding_dim)
    embed_rms = float(embed_tokens.weight.detach().float().pow(2).mean(dim=1).sqrt().median().cpu())
    anchor_ids = oracle_gate._encode_ids(tokenizer, anchor_text, device=resolved_device, add_special_tokens=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_items = _build_items(
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
        feature_mode=source_feature_mode,
    )
    eval_items = _build_items(
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
        feature_mode=source_feature_mode,
    )
    feature_dim = int(np.asarray(train_items[0]["source_features"]).shape[0])
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
    residual_encoder = SourceResidualSlotEncoder(
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        prefix_len=prefix_len,
        initial_residual_gate=initial_residual_gate,
    )
    residual_log = _train_source_residual_encoder(
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
                _source_packet_features(frozen_scores, feature_mode=source_feature_mode),
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
                "source_residual_slots": [
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
                "zero_source_residual": [
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
                "wrong_source_residual": [
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
                "candidate_roll_source_residual": [
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
                "target_derived_residual": [
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
                "candidate_derangement": [],
            }
            condition_scores["candidate_derangement"] = list(np.roll(condition_scores["source_residual_slots"], 1))
            _add_prediction_rows(prediction_rows=prediction_rows, item=item, condition_scores=condition_scores)
            row_summaries.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "answer_index": int(row.answer_index),
                    "full_prompt_prediction": int(_prediction(condition_scores["full_prompt"])),
                    "frozen_target_slots_prediction": int(_prediction(condition_scores["frozen_target_slots"])),
                    "source_residual_prediction": int(_prediction(condition_scores["source_residual_slots"])),
                    "source_top1_prediction": int(_prediction(condition_scores["source_top1_label_control"])),
                    "source_residual_kl_to_full": float(
                        _kl_to_full(condition_scores["source_residual_slots"], condition_scores["full_prompt"])
                    ),
                    "frozen_target_slots_kl_to_full": float(
                        _kl_to_full(condition_scores["frozen_target_slots"], condition_scores["full_prompt"])
                    ),
                    "source_residual_rms": float(source_residual.float().pow(2).mean().sqrt().cpu()),
                    "source_feature_norm": float(np.linalg.norm(np.asarray(item["source_features"], dtype=np.float64))),
                }
            )

    metrics = _condition_metrics(prediction_rows, seed=seed + 4242, bootstrap_samples=bootstrap_samples)
    answers = _answers_from_prediction_rows(prediction_rows)
    method_predictions = _predictions_for_condition(prediction_rows, "source_residual_slots")
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
    destructive_controls = (
        "zero_source_residual",
        "wrong_source_residual",
        "candidate_roll_source_residual",
        "target_derived_residual",
        "random_same_norm_residual",
        "source_top1_label_control",
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
    method = metrics["source_residual_slots"]
    frozen = metrics["frozen_target_slots"]
    kl_gain_vs_frozen = float(frozen["mean_kl_to_full"] - method["mean_kl_to_full"])
    pass_gate = bool(
        int(method["nonfinite_kl_count"]) == 0
        and int(method["nonfinite_score_row_count"]) == 0
        and float(paired_vs_frozen["mean_delta"]) >= float(min_delta_vs_frozen)
        and float(paired_vs_frozen["ci95_low"]) >= float(min_ci_low_vs_frozen)
        and kl_gain_vs_frozen >= float(min_kl_gain_vs_frozen)
        and method["mean_kl_to_full"] <= float(max_mean_kl)
        and method["accuracy"] >= metrics[best_destructive_by_accuracy]["accuracy"]
    )
    headline = {
        "source_residual_accuracy": float(method["accuracy"]),
        "source_residual_agreement": float(method["agreement_with_full_prompt"]),
        "source_residual_mean_kl": float(method["mean_kl_to_full"]),
        "source_residual_nonfinite_kl_count": int(method["nonfinite_kl_count"]),
        "source_residual_nonfinite_score_row_count": int(method["nonfinite_score_row_count"]),
        "frozen_target_slots_accuracy": float(frozen["accuracy"]),
        "frozen_target_slots_mean_kl": float(frozen["mean_kl_to_full"]),
        "source_top1_label_accuracy": float(metrics["source_top1_label_control"]["accuracy"]),
        "kl_gain_vs_frozen_target_slots": kl_gain_vs_frozen,
        "paired_vs_frozen_target_slots": paired_vs_frozen,
        "paired_vs_source_top1_label_control": paired_vs_source_label,
        "best_destructive_by_accuracy": best_destructive_by_accuracy,
        "best_destructive_accuracy": float(metrics[best_destructive_by_accuracy]["accuracy"]),
        "best_destructive_by_kl": best_destructive_by_kl,
        "best_destructive_mean_kl": float(metrics[best_destructive_by_kl]["mean_kl_to_full"]),
    }
    source_packet_raw_bytes = 2 if source_feature_mode == "top2_margin" else feature_dim * 2
    source_packet_framed_bytes = source_packet_raw_bytes + 3
    interpretation = (
        "The source-conditioned residual-slot gate passes this small held-out slice. This is still a "
        "Mac-local smoke gate, but it promotes adjacent slices and seed repeats with the same controls."
        if pass_gate
        else "The source-conditioned residual-slot gate does not pass this held-out slice. The result is "
        "still informative because it directly tests source-present residuals against zero-source, "
        "wrong-source, target-derived, and label-copy controls."
    )
    next_exact_gate = (
        "Repeat adjacent validation slices and seeds, then freeze a larger source-residual packet family."
        if pass_gate
        else "Inspect row-level failures and either add a quantized source-conditioned candidate repair head "
        "or move to a stronger residual-slot codebook/denoising interface."
    )
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
        "source_train_metadata": train_source_metadata,
        "source_eval_metadata": eval_source_metadata,
        "source_model_family": source_model_family,
        "source_feature_mode": source_feature_mode,
        "source_packet_raw_bytes": int(source_packet_raw_bytes),
        "source_packet_framed_bytes": int(source_packet_framed_bytes),
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "source_hidden_vector_exposed": False,
        "source_score_vector_exposed": bool(source_feature_mode == "score_z"),
        "source_score_summary_exposed": bool(source_feature_mode == "top2_margin"),
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
            "This is a Mac-local source-conditioned residual-slot probe. A paper claim requires adjacent "
            "slices, seed repeats, cross-family falsification, and native systems rows before widening."
        ),
        "interpretation": interpretation,
        "next_exact_gate": next_exact_gate,
    }
    json_path = output_dir / "target_self_resonance_hellaswag_source_residual_slot_gate.json"
    md_path = output_dir / "target_self_resonance_hellaswag_source_residual_slot_gate.md"
    predictions_path = output_dir / "predictions.jsonl"
    row_summary_path = output_dir / "row_summaries.csv"
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    _write_jsonl(predictions_path, prediction_rows)
    _write_csv(row_summary_path, row_summaries)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            "target_self_resonance_hellaswag_source_residual_slot_gate.json": _sha256_file(json_path),
            "target_self_resonance_hellaswag_source_residual_slot_gate.md": _sha256_file(md_path),
            "predictions.jsonl": _sha256_file(predictions_path),
            "row_summaries.csv": _sha256_file(row_summary_path),
        },
        "headline": headline,
        "pass_gate": pass_gate,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return payload


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
    parser.add_argument("--source-feature-mode", choices=("top2_margin", "score_z"), default="top2_margin")
    parser.add_argument("--min-delta-vs-frozen", type=float, default=0.0)
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
        source_feature_mode=args.source_feature_mode,
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
