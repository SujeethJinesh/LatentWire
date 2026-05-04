from __future__ import annotations

"""Source-conditioned oracle-prefix distillation gate for HellaSwag.

The target self-resonance oracle gate showed that per-example optimized soft
prefixes can recreate Qwen's full-prompt HellaSwag behavior. This gate asks the
next source-private question: can a compact source-hidden code predict a
target-native soft prefix on held-out rows?

The compressed target path receives:

    source-code(TinyLlama hidden summaries) -> learned Qwen soft prefix
    -> fixed anchor -> candidate continuation

The target model never receives the original context text in the compressed
path. This is a source-conditioned method gate, not a tiny-byte systems win yet:
the default projected source code is still tens of fp16 values.
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
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_target_self_resonance_hellaswag_chunk_encoder_gate as chunk_gate  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_oracle_distill_gate as oracle_distill  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_soft_prefix_gate as oracle_gate  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_source_hidden_residual_slot_gate as hidden_gate  # noqa: E402
from scripts import build_target_self_resonance_hellaswag_source_residual_slot_gate as residual_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/target_self_resonance_hellaswag_source_oracle_distill_gate_20260504_"
    "tiny_to_qwen05_train16_validation64_72"
)
DEFAULT_TRAIN_PATH = chunk_gate.DEFAULT_TRAIN_PATH
DEFAULT_EVAL_PATH = chunk_gate.DEFAULT_EVAL_PATH
DEFAULT_TARGET_MODEL = chunk_gate.DEFAULT_TARGET_MODEL
DEFAULT_SOURCE_MODEL = hidden_gate.DEFAULT_SOURCE_MODEL
DEFAULT_SOURCE_TRAIN_SCORE_CACHE = hidden_gate.DEFAULT_SOURCE_TRAIN_SCORE_CACHE
DEFAULT_SOURCE_EVAL_SCORE_CACHE = hidden_gate.DEFAULT_SOURCE_EVAL_SCORE_CACHE
DEFAULT_SOURCE_TRAIN_HIDDEN_CACHE = hidden_gate.DEFAULT_SOURCE_TRAIN_HIDDEN_CACHE
DEFAULT_SOURCE_EVAL_HIDDEN_CACHE = hidden_gate.DEFAULT_SOURCE_EVAL_HIDDEN_CACHE

CONDITIONS = (
    "full_prompt",
    "mean_oracle_slots",
    "source_oracle_distill_prefix",
    "zero_source_code",
    "wrong_source_code",
    "candidate_roll_source_code",
    "target_score_derived_code",
    "random_same_norm_prefix",
    "source_top1_label_control",
    "source_top1_or_top2_oracle",
    "candidate_derangement",
)

DESTRUCTIVE_CONTROLS = (
    "zero_source_code",
    "wrong_source_code",
    "candidate_roll_source_code",
    "target_score_derived_code",
    "random_same_norm_prefix",
    "source_top1_label_control",
    "candidate_derangement",
)


class SourceOraclePrefixEncoder(torch.nn.Module):
    """Map a compact source code to a target-native soft prefix."""

    def __init__(
        self,
        *,
        feature_dim: int,
        embed_dim: int,
        hidden_dim: int,
        prefix_len: int,
        base_prefix: torch.Tensor,
        initial_residual_gate: float = -2.0,
    ) -> None:
        super().__init__()
        if tuple(base_prefix.shape) != (int(prefix_len), int(embed_dim)):
            raise ValueError("base_prefix has wrong shape")
        self.prefix_len = int(prefix_len)
        self.embed_dim = int(embed_dim)
        self.register_buffer("base_prefix", base_prefix.detach().float().clone())
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(int(feature_dim)),
            torch.nn.Linear(int(feature_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(hidden_dim), int(prefix_len) * int(embed_dim), bias=False),
        )
        self.residual_gate = torch.nn.Parameter(torch.tensor(float(initial_residual_gate)))

    def forward(self, source_code: torch.Tensor) -> torch.Tensor:
        if source_code.dim() != 1:
            raise ValueError("source_code must be a 1D tensor")
        residual = self.net(source_code.float()).view(self.prefix_len, self.embed_dim)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=1.0, neginf=-1.0)
        return self.base_prefix.to(source_code.device) + torch.sigmoid(self.residual_gate) * torch.tanh(residual)


@dataclass(frozen=True)
class FeatureProjection:
    mean: np.ndarray
    components: np.ndarray
    scale: np.ndarray

    @property
    def code_dim(self) -> int:
        return int(self.components.shape[1])


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


def _prefix_rms_loss(prefix: torch.Tensor, *, embed_rms: float) -> torch.Tensor:
    return chunk_gate._prefix_rms_loss(prefix, embed_rms=embed_rms)


def _prefix_distill_loss(prefix: torch.Tensor, oracle_prefix: torch.Tensor, *, embed_rms: float) -> torch.Tensor:
    return oracle_distill._prefix_distill_loss(prefix, oracle_prefix, embed_rms=embed_rms)


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


def _fit_feature_projection(features: np.ndarray, *, code_dim: int) -> FeatureProjection:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2D matrix")
    mean = matrix.mean(axis=0, keepdims=True).astype(np.float32)
    centered = matrix - mean
    max_rank = min(centered.shape)
    actual_dim = min(max(int(code_dim), 1), max_rank)
    if float(np.linalg.norm(centered)) < 1e-8:
        components = np.eye(centered.shape[1], actual_dim, dtype=np.float32)
    else:
        _, _, vt = np.linalg.svd(centered.astype(np.float64), full_matrices=False)
        components = vt[:actual_dim].T.astype(np.float32)
    projected = centered @ components
    scale = projected.std(axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    return FeatureProjection(mean=mean.reshape(-1), components=components, scale=scale.reshape(-1))


def _project_feature(feature: np.ndarray, projection: FeatureProjection) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.shape[0] != projection.mean.shape[0]:
        raise ValueError("feature width does not match projection")
    code = ((feature - projection.mean) @ projection.components) / projection.scale
    return np.nan_to_num(code, nan=0.0, posinf=6.0, neginf=-6.0).astype(np.float32)


def _add_projected_codes(items: Sequence[dict[str, Any]], projection: FeatureProjection) -> None:
    for item in items:
        item["source_code"] = _project_feature(np.asarray(item["source_features"]), projection)
        item["candidate_roll_source_code"] = _project_feature(
            np.asarray(item["candidate_roll_source_features"]),
            projection,
        )


def _contrastive_kl_penalty(pos_kl: torch.Tensor, neg_kl: torch.Tensor, *, margin: float) -> torch.Tensor:
    return torch.nn.functional.softplus(torch.as_tensor(float(margin), device=pos_kl.device) + pos_kl - neg_kl)


def _train_source_encoder(
    *,
    encoder: SourceOraclePrefixEncoder,
    target_model: Any,
    embed_tokens: Any,
    train_items: Sequence[dict[str, Any]],
    anchor_ids: torch.Tensor,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    oracle_weight: float,
    norm_weight: float,
    contrastive_weight: float,
    contrastive_margin: float,
    embed_rms: float,
    length_normalize: bool,
    seed: int,
) -> dict[str, Any]:
    encoder.to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = random.Random(int(seed))
    losses: list[float] = []
    kls: list[float] = []
    negative_kls: list[float] = []
    prefix_losses: list[float] = []
    for epoch in range(int(epochs)):
        indices = list(range(len(train_items)))
        rng.shuffle(indices)
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_negative_kl = 0.0
        epoch_prefix = 0.0
        for idx in indices:
            item = train_items[idx]
            optimizer.zero_grad(set_to_none=True)
            source_code = torch.as_tensor(item["source_code"], device=device, dtype=torch.float32)
            oracle_prefix = item["oracle_prefix"].to(device=device, dtype=embed_tokens.weight.dtype)
            prefix = _normalize_prefix_rms(encoder(source_code).to(dtype=embed_tokens.weight.dtype), embed_rms=embed_rms)
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
            prefix_loss = _prefix_distill_loss(prefix, oracle_prefix, embed_rms=embed_rms)
            negative_kl = torch.zeros((), dtype=kl.dtype, device=kl.device)
            contrastive_loss = torch.zeros((), dtype=kl.dtype, device=kl.device)
            if float(contrastive_weight) > 0.0 and len(train_items) > 1:
                wrong_idx = rng.randrange(len(train_items) - 1)
                if wrong_idx >= idx:
                    wrong_idx += 1
                wrong_code = torch.as_tensor(train_items[wrong_idx]["source_code"], device=device, dtype=torch.float32)
                wrong_prefix = _normalize_prefix_rms(
                    encoder(wrong_code).to(dtype=embed_tokens.weight.dtype),
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
                contrastive_loss = _contrastive_kl_penalty(kl, negative_kl, margin=contrastive_margin)
            loss = (
                kl
                + float(oracle_weight) * prefix_loss
                + float(norm_weight) * _prefix_rms_loss(prefix, embed_rms=embed_rms)
                + float(contrastive_weight) * contrastive_loss
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite source-oracle loss at epoch={epoch} row_index={idx}")
            epoch_loss += float(loss.detach().cpu())
            epoch_kl += float(kl.detach().cpu())
            epoch_negative_kl += float(negative_kl.detach().cpu())
            epoch_prefix += float(prefix_loss.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
        denom = max(1, len(train_items))
        losses.append(epoch_loss / denom)
        kls.append(epoch_kl / denom)
        negative_kls.append(epoch_negative_kl / denom)
        prefix_losses.append(epoch_prefix / denom)
    encoder.eval()
    return {
        "loss_initial": float(losses[0]) if losses else 0.0,
        "loss_final": float(losses[-1]) if losses else 0.0,
        "kl_initial": float(kls[0]) if kls else 0.0,
        "kl_final": float(kls[-1]) if kls else 0.0,
        "negative_kl_initial": float(negative_kls[0]) if negative_kls else 0.0,
        "negative_kl_final": float(negative_kls[-1]) if negative_kls else 0.0,
        "prefix_loss_initial": float(prefix_losses[0]) if prefix_losses else 0.0,
        "prefix_loss_final": float(prefix_losses[-1]) if prefix_losses else 0.0,
        "residual_gate": float(torch.sigmoid(encoder.residual_gate.detach()).cpu()),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "oracle_weight": float(oracle_weight),
        "norm_weight": float(norm_weight),
        "contrastive_weight": float(contrastive_weight),
        "contrastive_margin": float(contrastive_margin),
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
        "# Target Self-Resonance HellaSwag Source-Oracle Distill Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- artifact: `{payload['artifact_dir']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- source model family: `{payload['source_model_family']}`",
        f"- target model: `{payload['target_model_path']}`",
        f"- train/eval rows: `{payload['train_row_count']}` / `{payload['eval_row_count']}`",
        f"- source raw feature fp16 bytes: `{payload['source_feature_raw_bytes_fp16']}`",
        f"- projected source code fp16 bytes: `{payload['source_code_raw_bytes_fp16']}`",
        f"- target prefix fp16 bytes: `{payload['prefix_raw_bytes_fp16']}`",
        "",
        "## Result",
        "",
        f"- source-oracle accuracy: `{metrics['source_oracle_distill_prefix']['accuracy']:.6f}`",
        f"- mean-oracle-slot accuracy: `{metrics['mean_oracle_slots']['accuracy']:.6f}`",
        f"- source top1/top2 oracle accuracy: `{metrics['source_top1_or_top2_oracle']['accuracy']:.6f}`",
        f"- source-oracle mean KL: `{metrics['source_oracle_distill_prefix']['mean_kl_to_full']:.6f}`",
        f"- mean-oracle-slot mean KL: `{metrics['mean_oracle_slots']['mean_kl_to_full']:.6f}`",
        f"- best destructive accuracy: `{headline['best_destructive_accuracy']:.6f}` (`{headline['best_destructive_by_accuracy']}`)",
        f"- paired CI95 low vs mean oracle slots: `{headline['paired_vs_mean_oracle_slots']['ci95_low']:.6f}`",
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
    source_code_dim: int,
    oracle_steps: int,
    oracle_lr: float,
    oracle_top1_weight: float,
    oracle_norm_weight: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    oracle_weight: float,
    norm_weight: float,
    contrastive_weight: float,
    contrastive_margin: float,
    initial_residual_gate: float,
    seed: int,
    device: str,
    dtype: str,
    max_length: int,
    anchor_text: str,
    continuation_mode: str,
    length_normalize: bool,
    hidden_feature_mode: str,
    min_delta_vs_mean_oracle: float,
    min_ci_low_vs_mean_oracle: float,
    min_kl_gain_vs_mean_oracle: float,
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
    resolved_device = oracle_gate._resolve_device(device)

    train_hidden, train_hidden_metadata = hidden_gate._source_hidden_for_rows(
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
    eval_hidden, eval_hidden_metadata = hidden_gate._source_hidden_for_rows(
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

    train_items = hidden_gate._build_items(
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
    eval_items = hidden_gate._build_items(
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
    train_items = oracle_distill._add_train_oracles(
        train_items=train_items,
        target_model=model,
        embed_tokens=embed_tokens,
        anchor_ids=anchor_ids,
        device=resolved_device,
        oracle_steps=oracle_steps,
        oracle_lr=oracle_lr,
        oracle_top1_weight=oracle_top1_weight,
        oracle_norm_weight=oracle_norm_weight,
        embed_rms=embed_rms,
        length_normalize=length_normalize,
    )

    raw_train_features = np.stack([np.asarray(item["source_features"], dtype=np.float32) for item in train_items], axis=0)
    projection = _fit_feature_projection(raw_train_features, code_dim=source_code_dim)
    _add_projected_codes(train_items, projection)
    _add_projected_codes(eval_items, projection)
    feature_dim = int(raw_train_features.shape[1])
    code_dim = int(projection.code_dim)
    score_feature_dim = int(hidden_gate._source_score_features(train_items[0]["source_scores"]).shape[0])
    mean_oracle_prefix = torch.stack([item["oracle_prefix"].float() for item in train_items], dim=0).mean(dim=0)
    source_encoder = SourceOraclePrefixEncoder(
        feature_dim=code_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        prefix_len=prefix_len,
        base_prefix=mean_oracle_prefix,
        initial_residual_gate=initial_residual_gate,
    )
    source_log = _train_source_encoder(
        encoder=source_encoder,
        target_model=model,
        embed_tokens=embed_tokens,
        train_items=train_items,
        anchor_ids=anchor_ids,
        device=resolved_device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        oracle_weight=oracle_weight,
        norm_weight=norm_weight,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        embed_rms=embed_rms,
        length_normalize=length_normalize,
        seed=seed,
    )

    prediction_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        for eval_index, item in enumerate(eval_items):
            row = item["row"]
            source_code = torch.as_tensor(item["source_code"], device=resolved_device, dtype=torch.float32)
            mean_prefix = _normalize_prefix_rms(mean_oracle_prefix.to(resolved_device, dtype=embed_tokens.weight.dtype), embed_rms=embed_rms)
            source_prefix = _normalize_prefix_rms(
                source_encoder(source_code).to(dtype=embed_tokens.weight.dtype),
                embed_rms=embed_rms,
            )
            zero_prefix = _normalize_prefix_rms(
                source_encoder(torch.zeros_like(source_code)).to(dtype=embed_tokens.weight.dtype),
                embed_rms=embed_rms,
            )
            wrong_item = eval_items[(eval_index + 1) % len(eval_items)]
            wrong_prefix = _normalize_prefix_rms(
                source_encoder(torch.as_tensor(wrong_item["source_code"], device=resolved_device, dtype=torch.float32)).to(
                    dtype=embed_tokens.weight.dtype
                ),
                embed_rms=embed_rms,
            )
            rolled_prefix = _normalize_prefix_rms(
                source_encoder(
                    torch.as_tensor(item["candidate_roll_source_code"], device=resolved_device, dtype=torch.float32)
                ).to(dtype=embed_tokens.weight.dtype),
                embed_rms=embed_rms,
            )
            mean_scores = [
                float(value)
                for value in _prefix_scores(
                    target_model=model,
                    embed_tokens=embed_tokens,
                    prefix=mean_prefix,
                    anchor_ids=anchor_ids,
                    choice_ids=item["choice_ids"],
                    device=resolved_device,
                    length_normalize=length_normalize,
                ).detach().cpu()
            ]
            target_features = hidden_gate._target_score_derived_features(
                frozen_scores=mean_scores,
                feature_dim=feature_dim,
                score_feature_dim=score_feature_dim,
            )
            target_code = torch.as_tensor(_project_feature(target_features, projection), device=resolved_device, dtype=torch.float32)
            target_prefix = _normalize_prefix_rms(
                source_encoder(target_code).to(dtype=embed_tokens.weight.dtype),
                embed_rms=embed_rms,
            )
            random_prefix = hidden_gate.chunk_gate._random_same_norm_prefix(
                reference=source_prefix,
                seed=seed * 1009 + eval_start + eval_index,
                device=resolved_device,
            ).to(resolved_device, dtype=source_prefix.dtype)
            condition_scores = {
                "full_prompt": [float(value) for value in item["full_scores"]],
                "mean_oracle_slots": mean_scores,
                "source_oracle_distill_prefix": [
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
                "zero_source_code": [
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
                "wrong_source_code": [
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
                "candidate_roll_source_code": [
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
                "target_score_derived_code": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=target_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "random_same_norm_prefix": [
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
                "source_top1_label_control": hidden_gate._source_top1_scores(item["source_scores"]),
                "source_top1_or_top2_oracle": hidden_gate._source_top1_or_top2_oracle_scores(
                    item["source_scores"],
                    row.answer_index,
                ),
                "candidate_derangement": [],
            }
            condition_scores["candidate_derangement"] = list(np.roll(condition_scores["source_oracle_distill_prefix"], 1))
            _add_prediction_rows(prediction_rows=prediction_rows, item=item, condition_scores=condition_scores)
            row_summaries.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "answer_index": int(row.answer_index),
                    "full_prompt_prediction": int(_prediction(condition_scores["full_prompt"])),
                    "mean_oracle_slots_prediction": int(_prediction(condition_scores["mean_oracle_slots"])),
                    "source_oracle_distill_prediction": int(
                        _prediction(condition_scores["source_oracle_distill_prefix"])
                    ),
                    "source_top1_prediction": int(_prediction(condition_scores["source_top1_label_control"])),
                    "source_top1_or_top2_oracle_prediction": int(
                        _prediction(condition_scores["source_top1_or_top2_oracle"])
                    ),
                    "source_oracle_kl_to_full": float(
                        _kl_to_full(condition_scores["source_oracle_distill_prefix"], condition_scores["full_prompt"])
                    ),
                    "mean_oracle_kl_to_full": float(
                        _kl_to_full(condition_scores["mean_oracle_slots"], condition_scores["full_prompt"])
                    ),
                    "source_code_norm": float(np.linalg.norm(np.asarray(item["source_code"], dtype=np.float64))),
                    "source_prefix_rms": float(source_prefix.float().pow(2).mean().sqrt().cpu()),
                }
            )

    metrics = _condition_metrics(prediction_rows, seed=seed + 4242, bootstrap_samples=bootstrap_samples)
    answers = _answers_from_prediction_rows(prediction_rows)
    method_predictions = _predictions_for_condition(prediction_rows, "source_oracle_distill_prefix")
    mean_predictions = _predictions_for_condition(prediction_rows, "mean_oracle_slots")
    source_label_predictions = _predictions_for_condition(prediction_rows, "source_top1_label_control")
    paired_vs_mean = oracle_gate._paired_ci(
        selected=method_predictions,
        baseline=mean_predictions,
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
    method = metrics["source_oracle_distill_prefix"]
    mean_control = metrics["mean_oracle_slots"]
    kl_gain_vs_mean = float(mean_control["mean_kl_to_full"] - method["mean_kl_to_full"])
    best_destructive_accuracy = float(metrics[best_destructive_by_accuracy]["accuracy"])
    pass_gate = bool(
        int(method["nonfinite_kl_count"]) == 0
        and int(method["nonfinite_score_row_count"]) == 0
        and float(paired_vs_mean["mean_delta"]) >= float(min_delta_vs_mean_oracle)
        and float(paired_vs_mean["ci95_low"]) >= float(min_ci_low_vs_mean_oracle)
        and kl_gain_vs_mean >= float(min_kl_gain_vs_mean_oracle)
        and method["mean_kl_to_full"] <= float(max_mean_kl)
        and float(method["accuracy"]) > best_destructive_accuracy
    )
    oracle_train_logs = [item["oracle_log"] for item in train_items]
    headline = {
        "source_oracle_accuracy": float(method["accuracy"]),
        "source_oracle_agreement": float(method["agreement_with_full_prompt"]),
        "source_oracle_mean_kl": float(method["mean_kl_to_full"]),
        "source_oracle_nonfinite_kl_count": int(method["nonfinite_kl_count"]),
        "source_oracle_nonfinite_score_row_count": int(method["nonfinite_score_row_count"]),
        "mean_oracle_slots_accuracy": float(mean_control["accuracy"]),
        "mean_oracle_slots_mean_kl": float(mean_control["mean_kl_to_full"]),
        "source_top1_label_accuracy": float(metrics["source_top1_label_control"]["accuracy"]),
        "source_top1_or_top2_oracle_accuracy": float(metrics["source_top1_or_top2_oracle"]["accuracy"]),
        "kl_gain_vs_mean_oracle_slots": kl_gain_vs_mean,
        "paired_vs_mean_oracle_slots": paired_vs_mean,
        "paired_vs_source_top1_label_control": paired_vs_source_label,
        "best_destructive_by_accuracy": best_destructive_by_accuracy,
        "best_destructive_accuracy": best_destructive_accuracy,
        "best_destructive_by_kl": best_destructive_by_kl,
        "best_destructive_mean_kl": float(metrics[best_destructive_by_kl]["mean_kl_to_full"]),
        "oracle_train_mean_kl_initial": float(statistics.fmean(log["kl_initial"] for log in oracle_train_logs)),
        "oracle_train_mean_kl_final": float(statistics.fmean(log["kl_final"] for log in oracle_train_logs)),
    }
    interpretation = (
        "The source-oracle distill gate passes this held-out slice. This should be treated as a live "
        "source-conditioned resonance branch and repeated across adjacent slices and seeds."
        if pass_gate
        else "The source-oracle distill gate does not pass this held-out slice. If the method ties or "
        "loses to wrong-source/zero-source controls, the current projected source code is not causal."
    )
    next_exact_gate = (
        "Repeat adjacent frozen slices with seed repeats, then quantize the projected source code and compare to same-byte text/code."
        if pass_gate
        else "Inspect row-level helps/harms, then try a stricter target-error-only objective or a smaller source-code ECC/syndrome branch."
    )
    source_feature_raw_bytes_fp16 = int(feature_dim) * 2
    source_code_raw_bytes_fp16 = int(code_dim) * 2
    payload: dict[str, Any] = {
        "date": run_date,
        "artifact_dir": _display(output_dir),
        "pass_gate": pass_gate,
        "headline": headline,
        "metrics": metrics,
        "source_log": source_log,
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
        "source_code_dim": int(code_dim),
        "source_feature_raw_bytes_fp16": int(source_feature_raw_bytes_fp16),
        "source_code_raw_bytes_fp16": int(source_code_raw_bytes_fp16),
        "source_text_exposed": False,
        "source_kv_exposed": False,
        "raw_source_hidden_exposed_to_target": False,
        "projected_source_code_exposed_to_bridge": True,
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
        "encoder_parameter_count": int(sum(param.numel() for param in source_encoder.parameters())),
        "embed_rms_median": float(embed_rms),
        "anchor_text": anchor_text,
        "anchor_token_count": int(anchor_ids.numel()),
        "continuation_mode": continuation_mode,
        "length_normalize": bool(length_normalize),
        "pass_criteria": {
            "min_delta_vs_mean_oracle": float(min_delta_vs_mean_oracle),
            "min_ci_low_vs_mean_oracle": float(min_ci_low_vs_mean_oracle),
            "min_kl_gain_vs_mean_oracle": float(min_kl_gain_vs_mean_oracle),
            "max_mean_kl": float(max_mean_kl),
        },
        "bootstrap_samples": int(bootstrap_samples),
        "runtime_s": float(time.perf_counter() - start_time),
        "peak_rss_mib": float(_peak_rss_mib()),
        "claim_boundary": (
            "This gate tests source-conditioned target-prefix prediction from projected hidden features. "
            "It is not a native systems result and not yet a fixed-byte packet protocol."
        ),
        "interpretation": interpretation,
        "next_exact_gate": next_exact_gate,
    }
    json_path = output_dir / "target_self_resonance_hellaswag_source_oracle_distill_gate.json"
    md_path = output_dir / "target_self_resonance_hellaswag_source_oracle_distill_gate.md"
    predictions_path = output_dir / "predictions.jsonl"
    row_summary_path = output_dir / "row_summaries.csv"
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    _write_jsonl(predictions_path, prediction_rows)
    _write_csv(row_summary_path, row_summaries)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            "target_self_resonance_hellaswag_source_oracle_distill_gate.json": _sha256_file(json_path),
            "target_self_resonance_hellaswag_source_oracle_distill_gate.md": _sha256_file(md_path),
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
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--eval-start", type=int, default=64)
    parser.add_argument("--eval-rows", type=int, default=8)
    parser.add_argument("--prefix-len", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--source-code-dim", type=int, default=64)
    parser.add_argument("--oracle-steps", type=int, default=24)
    parser.add_argument("--oracle-lr", type=float, default=0.005)
    parser.add_argument("--oracle-top1-weight", type=float, default=0.0)
    parser.add_argument("--oracle-norm-weight", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--oracle-weight", type=float, default=0.2)
    parser.add_argument("--norm-weight", type=float, default=0.001)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-margin", type=float, default=0.05)
    parser.add_argument("--initial-residual-gate", type=float, default=-2.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--anchor-text", default="Continuation:")
    parser.add_argument("--continuation-mode", choices=("choice", "label_and_choice", "label"), default="choice")
    parser.add_argument("--length-normalize", choices=("true", "false"), default="true")
    parser.add_argument("--hidden-feature-mode", choices=("top2_delta", "mean_top1_delta"), default="top2_delta")
    parser.add_argument("--min-delta-vs-mean-oracle", type=float, default=0.001)
    parser.add_argument("--min-ci-low-vs-mean-oracle", type=float, default=0.0)
    parser.add_argument("--min-kl-gain-vs-mean-oracle", type=float, default=0.0)
    parser.add_argument("--max-mean-kl", type=float, default=0.25)
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
        source_code_dim=args.source_code_dim,
        oracle_steps=args.oracle_steps,
        oracle_lr=args.oracle_lr,
        oracle_top1_weight=args.oracle_top1_weight,
        oracle_norm_weight=args.oracle_norm_weight,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        oracle_weight=args.oracle_weight,
        norm_weight=args.norm_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        initial_residual_gate=args.initial_residual_gate,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        anchor_text=args.anchor_text,
        continuation_mode=args.continuation_mode,
        length_normalize=args.length_normalize == "true",
        hidden_feature_mode=args.hidden_feature_mode,
        min_delta_vs_mean_oracle=args.min_delta_vs_mean_oracle,
        min_ci_low_vs_mean_oracle=args.min_ci_low_vs_mean_oracle,
        min_kl_gain_vs_mean_oracle=args.min_kl_gain_vs_mean_oracle,
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
