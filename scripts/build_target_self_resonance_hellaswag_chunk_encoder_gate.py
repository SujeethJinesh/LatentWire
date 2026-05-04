from __future__ import annotations

"""Held-out target self-resonance gate with a learned chunk-prefix encoder.

The previous target self-resonance gate optimized one soft prefix per validation
example.  This gate is stricter: it trains a small shared encoder on official
HellaSwag train rows, freezes it, and evaluates held-out validation rows.  The
target model never receives the original context text in the compressed path;
it receives learned soft slots plus a fixed anchor and candidate continuations.
"""

import argparse
import csv
import datetime as dt
import hashlib
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

from scripts import build_target_self_resonance_hellaswag_soft_prefix_gate as oracle_gate  # noqa: E402
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/target_self_resonance_hellaswag_chunk_encoder_gate_20260504_qwen05_train32_validation32_48"
)
DEFAULT_TRAIN_PATH = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_train.jsonl"
)
DEFAULT_EVAL_PATH = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation.jsonl"
)
DEFAULT_TARGET_MODEL = oracle_gate.DEFAULT_TARGET_MODEL

CONDITIONS = (
    "full_prompt",
    "chunk_mean_prefix",
    "learned_chunk_encoder",
    "slots_only_encoder",
    "zero_prefix",
    "random_same_norm_prefix",
    "shuffled_chunk_encoder",
    "candidate_derangement",
)


class ChunkPrefixEncoder(torch.nn.Module):
    def __init__(self, *, embed_dim: int, hidden_dim: int, prefix_len: int) -> None:
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.norm = torch.nn.LayerNorm(int(embed_dim))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(embed_dim), int(hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Linear(int(hidden_dim), int(embed_dim)),
        )
        self.bias = torch.nn.Parameter(torch.zeros(int(prefix_len), int(embed_dim)))
        self.residual_gate = torch.nn.Parameter(torch.tensor(-4.0))

    def forward(self, chunk_prefix: torch.Tensor) -> torch.Tensor:
        if chunk_prefix.shape[0] != self.prefix_len:
            raise ValueError("chunk_prefix has wrong prefix length")
        residual = self.net(self.norm(chunk_prefix))
        return chunk_prefix + torch.sigmoid(self.residual_gate) * residual + self.bias


class SlotsOnlyEncoder(torch.nn.Module):
    def __init__(self, *, initial_prefix: torch.Tensor) -> None:
        super().__init__()
        self.slots = torch.nn.Parameter(initial_prefix.detach().clone())

    def forward(self, chunk_prefix: torch.Tensor) -> torch.Tensor:
        del chunk_prefix
        return self.slots


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    return oracle_gate._resolve(path)


def _display(path: pathlib.Path | str) -> str:
    return oracle_gate._display(path)


def _sha256_file(path: pathlib.Path | str) -> str:
    return oracle_gate._sha256_file(path)


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    oracle_gate._write_json(path, payload)


def _write_jsonl(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    oracle_gate._write_jsonl(path, rows)


def _write_csv(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    oracle_gate._write_csv(path, rows)


def _peak_rss_mib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def _resolve_device(device: str) -> str:
    return oracle_gate._resolve_device(device)


def _torch_dtype(dtype: str) -> Any:
    return oracle_gate._torch_dtype(dtype)


def _prediction(scores: Sequence[float] | torch.Tensor) -> int:
    if isinstance(scores, torch.Tensor):
        values = [float(value) for value in scores.detach().cpu()]
    else:
        values = [float(value) for value in scores]
    finite_values = [value if math.isfinite(value) else float("-inf") for value in values]
    if not finite_values or all(value == float("-inf") for value in finite_values):
        return 0
    return int(max(range(len(finite_values)), key=lambda index: (finite_values[index], -index)))


def _kl_to_full(condition_scores: Sequence[float], full_scores: Sequence[float]) -> float:
    return oracle_gate._kl_to_full(condition_scores, full_scores)


def _margin(scores: Sequence[float], answer_index: int) -> float:
    return oracle_gate._margin(scores, answer_index)


def _row_slice(rows: Sequence[arc_gate.ArcRow], *, start: int, limit: int) -> list[arc_gate.ArcRow]:
    selected = list(rows[int(start) : int(start) + int(limit)])
    if not selected:
        raise ValueError("selected no rows")
    return selected


def _encode_row(
    *,
    row: arc_gate.ArcRow,
    tokenizer: Any,
    embed_tokens: Any,
    target_model: Any,
    device: str,
    prefix_len: int,
    embed_rms: float,
    max_length: int,
    continuation_mode: str,
    length_normalize: bool,
) -> dict[str, Any]:
    prompt_ids = oracle_gate._encode_ids(
        tokenizer,
        oracle_gate._prompt(row),
        device=device,
        add_special_tokens=True,
    )[-int(max_length) :]
    choice_ids = [
        oracle_gate._encode_ids(
            tokenizer,
            oracle_gate._continuation(row, index, mode=continuation_mode),
            device=device,
            add_special_tokens=False,
        )
        for index, _ in enumerate(row.choices)
    ]
    with torch.no_grad():
        full_scores = oracle_gate._full_prompt_scores(
            target_model=target_model,
            embed_tokens=embed_tokens,
            prompt_ids=prompt_ids,
            continuation_ids=choice_ids,
            length_normalize=length_normalize,
        )
        chunk_prefix = oracle_gate._prompt_chunk_mean_prefix(
            prompt_ids=prompt_ids,
            embed_tokens=embed_tokens,
            prefix_len=prefix_len,
            embed_rms=embed_rms,
            device=device,
        )
    return {
        "row": row,
        "prompt_ids": prompt_ids.detach().cpu(),
        "choice_ids": [ids.detach().cpu() for ids in choice_ids],
        "full_scores": full_scores.detach().cpu(),
        "full_probs": torch.softmax(full_scores.detach().float().cpu(), dim=-1),
        "chunk_prefix": chunk_prefix.detach().cpu(),
    }


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
    prefix = prefix.to(device=device, dtype=embed_tokens.weight.dtype)
    context = oracle_gate._prefix_context(
        prefix=prefix,
        anchor_ids=anchor_ids,
        embed_tokens=embed_tokens,
    )
    return oracle_gate._choice_scores_from_context(
        target_model=target_model,
        embed_tokens=embed_tokens,
        context_embeds=context,
        continuation_ids=[ids.to(device) for ids in choice_ids],
        length_normalize=length_normalize,
    )


def _prefix_rms_loss(prefix: torch.Tensor, *, embed_rms: float) -> torch.Tensor:
    token_rms = prefix.float().pow(2).mean(dim=1).sqrt()
    return torch.mean((token_rms - float(embed_rms)) ** 2)


def _normalize_prefix_rms(prefix: torch.Tensor, *, embed_rms: float) -> torch.Tensor:
    row_rms = prefix.float().pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    normalized = prefix.float() / row_rms * float(embed_rms)
    return normalized.to(dtype=prefix.dtype, device=prefix.device)


def _train_encoder(
    *,
    encoder: torch.nn.Module,
    target_model: Any,
    embed_tokens: Any,
    train_items: Sequence[dict[str, Any]],
    anchor_ids: torch.Tensor,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    norm_weight: float,
    embed_rms: float,
    length_normalize: bool,
    seed: int,
) -> dict[str, Any]:
    encoder.to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    rng = random.Random(int(seed))
    losses: list[float] = []
    kls: list[float] = []
    for epoch in range(int(epochs)):
        indices = list(range(len(train_items)))
        rng.shuffle(indices)
        epoch_loss = 0.0
        epoch_kl = 0.0
        for idx in indices:
            item = train_items[idx]
            optimizer.zero_grad(set_to_none=True)
            chunk_prefix = item["chunk_prefix"].to(device=device, dtype=embed_tokens.weight.dtype)
            prefix = _normalize_prefix_rms(encoder(chunk_prefix), embed_rms=embed_rms)
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
            loss = kl + float(norm_weight) * _prefix_rms_loss(prefix, embed_rms=embed_rms)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"nonfinite encoder loss at epoch={epoch} row_index={idx}: "
                    f"loss={float(loss.detach().cpu())} kl={float(kl.detach().cpu())}"
                )
            epoch_loss += float(loss.detach().cpu())
            epoch_kl += float(kl.detach().cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
        losses.append(epoch_loss / max(1, len(train_items)))
        kls.append(epoch_kl / max(1, len(train_items)))
    encoder.eval()
    return {
        "loss_initial": float(losses[0]) if losses else 0.0,
        "loss_final": float(losses[-1]) if losses else 0.0,
        "kl_initial": float(kls[0]) if kls else 0.0,
        "kl_final": float(kls[-1]) if kls else 0.0,
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "norm_weight": float(norm_weight),
    }


def _random_same_norm_prefix(*, reference: torch.Tensor, seed: int, device: str) -> torch.Tensor:
    prefix = oracle_gate._random_same_norm_prefix(reference=reference.to(device), seed=seed)
    return prefix.to(reference.device)


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
        "# Target Self-Resonance HellaSwag Chunk-Encoder Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- artifact: `{payload['artifact_dir']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- target model: `{payload['target_model_path']}`",
        f"- train/eval rows: `{payload['train_row_count']}` / `{payload['eval_row_count']}`",
        f"- prefix tokens: `{payload['prefix_len']}`",
        "",
        "## Result",
        "",
        f"- learned encoder agreement: `{metrics['learned_chunk_encoder']['agreement_with_full_prompt']:.6f}`",
        f"- learned encoder mean KL: `{metrics['learned_chunk_encoder']['mean_kl_to_full']:.6f}`",
        f"- chunk-mean agreement/KL: `{metrics['chunk_mean_prefix']['agreement_with_full_prompt']:.6f}` / `{metrics['chunk_mean_prefix']['mean_kl_to_full']:.6f}`",
        f"- slots-only agreement/KL: `{metrics['slots_only_encoder']['agreement_with_full_prompt']:.6f}` / `{metrics['slots_only_encoder']['mean_kl_to_full']:.6f}`",
        f"- best destructive agreement: `{headline['best_destructive_agreement']:.6f}` (`{headline['best_destructive_by_agreement']}`)",
        f"- best destructive mean KL: `{headline['best_destructive_mean_kl']:.6f}` (`{headline['best_destructive_by_kl']}`)",
        "",
        "## Condition Metrics",
        "",
        "| Condition | Accuracy | Full agreement | Mean KL | Mean margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = metrics[condition]
        lines.append(
            f"| `{condition}` | {row['accuracy']:.6f} | {row['agreement_with_full_prompt']:.6f} | "
            f"{row['mean_kl_to_full']:.6f} | {row['mean_margin']:.6f} |"
        )
    lines.extend(["", "## Interpretation", "", payload["interpretation"], "", "## Next Gate", "", payload["next_exact_gate"], ""])
    _resolve(path).parent.mkdir(parents=True, exist_ok=True)
    _resolve(path).write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path,
    train_path: pathlib.Path,
    eval_path: pathlib.Path,
    target_model_path: str,
    train_start: int,
    train_rows: int,
    eval_start: int,
    eval_rows: int,
    prefix_len: int,
    hidden_dim: int,
    epochs: int,
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
    min_kl_gain_vs_chunk: float,
    min_kl_gain_vs_slots: float,
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
    all_train_rows = arc_gate._load_rows(train_path)
    all_eval_rows = arc_gate._load_rows(eval_path)
    selected_train = _row_slice(all_train_rows, start=train_start, limit=train_rows)
    selected_eval = _row_slice(all_eval_rows, start=eval_start, limit=eval_rows)
    train_content = {row.content_id for row in selected_train}
    eval_content = {row.content_id for row in selected_eval}
    content_overlap = sorted(train_content & eval_content)
    if content_overlap:
        raise ValueError(f"train/eval content overlap: {content_overlap[:3]}")
    resolved_device = _resolve_device(device)

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
        torch_dtype=_torch_dtype(dtype),
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

    train_items = [
        _encode_row(
            row=row,
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            target_model=model,
            device=resolved_device,
            prefix_len=prefix_len,
            embed_rms=embed_rms,
            max_length=max_length,
            continuation_mode=continuation_mode,
            length_normalize=length_normalize,
        )
        for row in selected_train
    ]
    eval_items = [
        _encode_row(
            row=row,
            tokenizer=tokenizer,
            embed_tokens=embed_tokens,
            target_model=model,
            device=resolved_device,
            prefix_len=prefix_len,
            embed_rms=embed_rms,
            max_length=max_length,
            continuation_mode=continuation_mode,
            length_normalize=length_normalize,
        )
        for row in selected_eval
    ]
    train_mean_prefix = torch.stack([item["chunk_prefix"].float() for item in train_items], dim=0).mean(dim=0)
    learned_encoder = ChunkPrefixEncoder(embed_dim=embed_dim, hidden_dim=hidden_dim, prefix_len=prefix_len)
    slots_encoder = SlotsOnlyEncoder(initial_prefix=train_mean_prefix.to(dtype=embed_tokens.weight.dtype))
    learned_log = _train_encoder(
        encoder=learned_encoder,
        target_model=model,
        embed_tokens=embed_tokens,
        train_items=train_items,
        anchor_ids=anchor_ids,
        device=resolved_device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        norm_weight=norm_weight,
        embed_rms=embed_rms,
        length_normalize=length_normalize,
        seed=seed,
    )
    slots_log = _train_encoder(
        encoder=slots_encoder,
        target_model=model,
        embed_tokens=embed_tokens,
        train_items=train_items,
        anchor_ids=anchor_ids,
        device=resolved_device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        norm_weight=norm_weight,
        embed_rms=embed_rms,
        length_normalize=length_normalize,
        seed=seed + 100,
    )

    prediction_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []
    eval_chunk_prefixes = [item["chunk_prefix"].to(resolved_device, dtype=embed_tokens.weight.dtype) for item in eval_items]
    with torch.no_grad():
        for eval_index, item in enumerate(eval_items):
            row = item["row"]
            chunk_prefix = eval_chunk_prefixes[eval_index]
            learned_prefix = _normalize_prefix_rms(learned_encoder(chunk_prefix), embed_rms=embed_rms)
            slots_prefix = _normalize_prefix_rms(slots_encoder(chunk_prefix), embed_rms=embed_rms)
            zero_prefix = torch.zeros_like(learned_prefix)
            random_prefix = _random_same_norm_prefix(
                reference=learned_prefix,
                seed=seed * 1009 + eval_start + eval_index,
                device=resolved_device,
            ).to(resolved_device, dtype=learned_prefix.dtype)
            shuffled_prefix = _normalize_prefix_rms(
                learned_encoder(eval_chunk_prefixes[(eval_index + 1) % len(eval_chunk_prefixes)]),
                embed_rms=embed_rms,
            )
            condition_scores = {
                "full_prompt": [float(value) for value in item["full_scores"]],
                "chunk_mean_prefix": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=chunk_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "learned_chunk_encoder": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=learned_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "slots_only_encoder": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=slots_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "zero_prefix": [
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
                "shuffled_chunk_encoder": [
                    float(value)
                    for value in _prefix_scores(
                        target_model=model,
                        embed_tokens=embed_tokens,
                        prefix=shuffled_prefix,
                        anchor_ids=anchor_ids,
                        choice_ids=item["choice_ids"],
                        device=resolved_device,
                        length_normalize=length_normalize,
                    ).detach().cpu()
                ],
                "candidate_derangement": [],
            }
            condition_scores["candidate_derangement"] = list(np.roll(condition_scores["learned_chunk_encoder"], 1))
            _add_prediction_rows(
                prediction_rows=prediction_rows,
                item=item,
                condition_scores=condition_scores,
            )
            learned_kl = _kl_to_full(condition_scores["learned_chunk_encoder"], condition_scores["full_prompt"])
            chunk_kl = _kl_to_full(condition_scores["chunk_mean_prefix"], condition_scores["full_prompt"])
            learned_rms = float(learned_prefix.float().pow(2).mean().sqrt().cpu())
            chunk_rms = float(chunk_prefix.float().pow(2).mean().sqrt().cpu())
            row_summaries.append(
                {
                    "row_id": row.row_id,
                    "content_id": row.content_id,
                    "answer_index": int(row.answer_index),
                    "full_prompt_prediction": int(_prediction(condition_scores["full_prompt"])),
                    "learned_prediction": int(_prediction(condition_scores["learned_chunk_encoder"])),
                    "chunk_prediction": int(_prediction(condition_scores["chunk_mean_prefix"])),
                    "learned_kl_to_full": float(learned_kl if math.isfinite(learned_kl) else 1.0e9),
                    "chunk_kl_to_full": float(chunk_kl if math.isfinite(chunk_kl) else 1.0e9),
                    "learned_prefix_rms": float(learned_rms if math.isfinite(learned_rms) else 1.0e9),
                    "chunk_prefix_rms": float(chunk_rms if math.isfinite(chunk_rms) else 1.0e9),
                }
            )

    metrics = _condition_metrics(
        prediction_rows,
        seed=seed + 4242,
        bootstrap_samples=bootstrap_samples,
    )
    destructive_controls = (
        "slots_only_encoder",
        "zero_prefix",
        "random_same_norm_prefix",
        "shuffled_chunk_encoder",
        "candidate_derangement",
    )
    learned = metrics["learned_chunk_encoder"]
    chunk = metrics["chunk_mean_prefix"]
    slots = metrics["slots_only_encoder"]
    best_destructive_by_agreement = max(
        destructive_controls,
        key=lambda condition: metrics[condition]["agreement_with_full_prompt"],
    )
    best_destructive_by_kl = min(
        destructive_controls,
        key=lambda condition: (
            int(metrics[condition]["nonfinite_kl_count"]),
            int(metrics[condition]["nonfinite_score_row_count"]),
            float(metrics[condition]["mean_kl_to_full"]),
        ),
    )
    kl_gain_vs_chunk = float(chunk["mean_kl_to_full"] - learned["mean_kl_to_full"])
    kl_gain_vs_slots = float(slots["mean_kl_to_full"] - learned["mean_kl_to_full"])
    pass_gate = bool(
        int(learned["nonfinite_kl_count"]) == 0
        and int(learned["nonfinite_score_row_count"]) == 0
        and
        learned["agreement_with_full_prompt"] >= chunk["agreement_with_full_prompt"]
        and kl_gain_vs_chunk >= float(min_kl_gain_vs_chunk)
        and kl_gain_vs_slots >= float(min_kl_gain_vs_slots)
        and learned["mean_kl_to_full"] <= float(max_mean_kl)
    )
    headline = {
        "learned_accuracy": float(learned["accuracy"]),
        "learned_agreement": float(learned["agreement_with_full_prompt"]),
        "learned_mean_kl": float(learned["mean_kl_to_full"]),
        "learned_nonfinite_kl_count": int(learned["nonfinite_kl_count"]),
        "learned_nonfinite_score_row_count": int(learned["nonfinite_score_row_count"]),
        "chunk_mean_agreement": float(chunk["agreement_with_full_prompt"]),
        "chunk_mean_kl": float(chunk["mean_kl_to_full"]),
        "slots_only_agreement": float(slots["agreement_with_full_prompt"]),
        "slots_only_kl": float(slots["mean_kl_to_full"]),
        "kl_gain_vs_chunk_mean": kl_gain_vs_chunk,
        "kl_gain_vs_slots_only": kl_gain_vs_slots,
        "best_destructive_by_agreement": best_destructive_by_agreement,
        "best_destructive_agreement": float(metrics[best_destructive_by_agreement]["agreement_with_full_prompt"]),
        "best_destructive_by_kl": best_destructive_by_kl,
        "best_destructive_mean_kl": float(metrics[best_destructive_by_kl]["mean_kl_to_full"]),
    }
    interpretation = (
        "The held-out target-side encoder passes this small gate: it improves the target full-prompt "
        "distribution match over raw chunk means and slots-only controls without receiving the context "
        "text at target inference time. This promotes the next source-conditioned slot-population gate."
        if pass_gate
        else "The held-out target-side encoder does not yet pass. The oracle self-resonance capacity "
        "result remains alive, but the current shared chunk-residual encoder is not sufficient."
    )
    next_exact_gate = (
        "Add a source-conditioned residual into the same target slots and score source-present minus "
        "zero-source-slot, with shuffled-source and candidate-deranged controls."
        if pass_gate
        else "Try a stronger query-resampler/ICAE-style encoder or supervised distillation from oracle "
        "prefixes before returning to cross-model transfer."
    )
    payload: dict[str, Any] = {
        "date": run_date,
        "artifact_dir": _display(output_dir),
        "pass_gate": pass_gate,
        "headline": headline,
        "metrics": metrics,
        "train_log": learned_log,
        "slots_log": slots_log,
        "row_summaries": row_summaries,
        "train_path": _display(train_path),
        "train_sha256": _sha256_file(train_path),
        "eval_path": _display(eval_path),
        "eval_sha256": _sha256_file(eval_path),
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
        "encoder_parameter_count": int(sum(param.numel() for param in learned_encoder.parameters())),
        "slots_parameter_count": int(sum(param.numel() for param in slots_encoder.parameters())),
        "embed_rms_median": float(embed_rms),
        "anchor_text": anchor_text,
        "anchor_token_count": int(anchor_ids.numel()),
        "continuation_mode": continuation_mode,
        "length_normalize": bool(length_normalize),
        "pass_criteria": {
            "min_kl_gain_vs_chunk": float(min_kl_gain_vs_chunk),
            "min_kl_gain_vs_slots": float(min_kl_gain_vs_slots),
            "max_mean_kl": float(max_mean_kl),
        },
        "bootstrap_samples": int(bootstrap_samples),
        "runtime_s": float(time.perf_counter() - start_time),
        "peak_rss_mib": float(_peak_rss_mib()),
        "claim_boundary": (
            "This is target-side context compression into learned soft slots. It is not cross-model "
            "communication until a source-conditioned encoder adds held-out gain over the zero-source "
            "target-slot baseline."
        ),
        "interpretation": interpretation,
        "next_exact_gate": next_exact_gate,
    }
    json_path = output_dir / "target_self_resonance_hellaswag_chunk_encoder_gate.json"
    md_path = output_dir / "target_self_resonance_hellaswag_chunk_encoder_gate.md"
    predictions_path = output_dir / "predictions.jsonl"
    row_summary_path = output_dir / "row_summaries.csv"
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    _write_jsonl(predictions_path, prediction_rows)
    _write_csv(row_summary_path, row_summaries)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            "target_self_resonance_hellaswag_chunk_encoder_gate.json": _sha256_file(json_path),
            "target_self_resonance_hellaswag_chunk_encoder_gate.md": _sha256_file(md_path),
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
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-rows", type=int, default=32)
    parser.add_argument("--eval-start", type=int, default=32)
    parser.add_argument("--eval-rows", type=int, default=16)
    parser.add_argument("--prefix-len", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--norm-weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--anchor-text", default="Continuation:")
    parser.add_argument("--continuation-mode", choices=("choice", "label_and_choice", "label"), default="choice")
    parser.add_argument("--length-normalize", choices=("true", "false"), default="true")
    parser.add_argument("--min-kl-gain-vs-chunk", type=float, default=0.0)
    parser.add_argument("--min-kl-gain-vs-slots", type=float, default=0.0)
    parser.add_argument("--max-mean-kl", type=float, default=0.15)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--run-date", default=str(dt.date.today()))
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        train_path=args.train_path,
        eval_path=args.eval_path,
        target_model_path=args.target_model_path,
        train_start=args.train_start,
        train_rows=args.train_rows,
        eval_start=args.eval_start,
        eval_rows=args.eval_rows,
        prefix_len=args.prefix_len,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        norm_weight=args.norm_weight,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        anchor_text=args.anchor_text,
        continuation_mode=args.continuation_mode,
        length_normalize=args.length_normalize == "true",
        min_kl_gain_vs_chunk=args.min_kl_gain_vs_chunk,
        min_kl_gain_vs_slots=args.min_kl_gain_vs_slots,
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
