from __future__ import annotations

"""Target self-resonance capacity probe for HellaSwag soft prefixes.

This is a diagnostic gate, not yet a source-private cross-model method.  It
asks whether a frozen target LM's full-text multiple-choice behavior can be
recreated when the context is removed and replaced by a compact learned
continuous prefix.  A pass here keeps the target-resonance branch alive; a fail
would weaken full cross-model latent communication before we spend larger GPU
cycles on an encoder.
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

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path(
    "results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation0_16"
)
DEFAULT_EVAL_PATH = pathlib.Path(
    "results/source_private_hellaswag_bridge_contract_20260501/official_splits/hellaswag_validation.jsonl"
)
DEFAULT_TARGET_MODEL = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
CONDITIONS = (
    "full_prompt",
    "chunk_mean_prefix",
    "optimized_soft_prefix",
    "zero_prefix",
    "random_same_norm_prefix",
    "shuffled_optimized_prefix",
    "candidate_derangement",
)


def _resolve(path: pathlib.Path | str) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path | str) -> str:
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


def _write_json(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: pathlib.Path | str, rows: Sequence[dict[str, Any]]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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


def _torch_dtype(dtype: str) -> Any:
    return arc_gate._torch_dtype(dtype)


def _resolve_device(device: str) -> str:
    return "cpu" if device == "auto_cpu" else arc_gate.syn._resolve_torch_device(device)


def _prompt(row: arc_gate.ArcRow) -> str:
    return f"Choose the most plausible continuation.\nContext: {row.question}\nContinuation:"


def _continuation(row: arc_gate.ArcRow, choice_index: int, *, mode: str) -> str:
    if mode == "choice":
        return f" {row.choices[choice_index]}"
    if mode == "label_and_choice":
        return f" {row.choice_labels[choice_index]}. {row.choices[choice_index]}"
    if mode == "label":
        return f" {row.choice_labels[choice_index]}"
    raise ValueError(f"unknown continuation mode {mode!r}")


def _encode_ids(tokenizer: Any, text: str, *, device: str, add_special_tokens: bool) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    ids = encoded.input_ids[0].to(device)
    if ids.numel() == 0:
        raise ValueError(f"zero-token text: {text!r}")
    return ids


def _choice_scores_from_context(
    *,
    target_model: Any,
    embed_tokens: Any,
    context_embeds: torch.Tensor,
    continuation_ids: Sequence[torch.Tensor],
    length_normalize: bool,
) -> torch.Tensor:
    if context_embeds.dim() != 2 or context_embeds.shape[0] < 1:
        raise ValueError("context_embeds must be [context_len, embed_dim] with context_len >= 1")
    if not continuation_ids:
        raise ValueError("continuation_ids must not be empty")
    device = context_embeds.device
    dtype = context_embeds.dtype
    batched_inputs: list[torch.Tensor] = []
    input_lengths: list[int] = []
    continuation_tensors: list[torch.Tensor] = []
    for ids in continuation_ids:
        ids = ids.to(device)
        continuation_tensors.append(ids)
        continuation_embeds = embed_tokens(ids).detach().to(device=device, dtype=dtype)
        if continuation_embeds.shape[0] > 1:
            inputs = torch.cat([context_embeds, continuation_embeds[:-1]], dim=0)
        else:
            inputs = context_embeds
        batched_inputs.append(inputs)
        input_lengths.append(int(inputs.shape[0]))

    inputs_embeds = torch.nn.utils.rnn.pad_sequence(
        batched_inputs,
        batch_first=True,
        padding_value=0.0,
    )
    lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
    positions = torch.arange(inputs_embeds.shape[1], dtype=torch.long, device=device)
    attention_mask = (positions.unsqueeze(0) < lengths.unsqueeze(1)).long()
    out = target_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
    logits = out.logits
    start = int(context_embeds.shape[0] - 1)
    scores: list[torch.Tensor] = []
    for choice_index, ids in enumerate(continuation_tensors):
        token_logits = logits[choice_index, start : start + ids.shape[0]]
        logprobs = torch.log_softmax(token_logits.float(), dim=-1)
        score = logprobs.gather(1, ids[:, None]).sum()
        if length_normalize:
            score = score / max(int(ids.shape[0]), 1)
        scores.append(score)
    return torch.stack(scores)


def _full_prompt_scores(
    *,
    target_model: Any,
    embed_tokens: Any,
    prompt_ids: torch.Tensor,
    continuation_ids: Sequence[torch.Tensor],
    length_normalize: bool,
) -> torch.Tensor:
    prompt_embeds = embed_tokens(prompt_ids).detach()
    return _choice_scores_from_context(
        target_model=target_model,
        embed_tokens=embed_tokens,
        context_embeds=prompt_embeds,
        continuation_ids=continuation_ids,
        length_normalize=length_normalize,
    )


def _prefix_context(
    *,
    prefix: torch.Tensor,
    anchor_ids: torch.Tensor,
    embed_tokens: Any,
) -> torch.Tensor:
    anchor_embeds = embed_tokens(anchor_ids.to(prefix.device)).detach().to(
        device=prefix.device,
        dtype=prefix.dtype,
    )
    return torch.cat([prefix, anchor_embeds], dim=0)


def _prompt_chunk_mean_prefix(
    *,
    prompt_ids: torch.Tensor,
    embed_tokens: Any,
    prefix_len: int,
    embed_rms: float,
    device: str,
) -> torch.Tensor:
    prompt_embeds = embed_tokens(prompt_ids.to(device)).detach().float()
    chunks = torch.chunk(prompt_embeds, int(prefix_len), dim=0)
    rows = []
    for chunk in chunks:
        row = chunk.mean(dim=0) if chunk.numel() else prompt_embeds.mean(dim=0)
        row_rms = row.float().pow(2).mean().sqrt().clamp_min(1e-8)
        rows.append(row / row_rms * float(embed_rms))
    while len(rows) < int(prefix_len):
        rows.append(rows[-1].clone())
    return torch.stack(rows[: int(prefix_len)], dim=0).to(device=device, dtype=embed_tokens.weight.dtype)


def _random_same_norm_prefix(
    *,
    reference: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(tuple(reference.shape), generator=generator, dtype=torch.float32).to(reference.device)
    ref_norms = reference.float().norm(dim=1, keepdim=True).clamp_min(1e-8)
    noise = noise / noise.float().norm(dim=1, keepdim=True).clamp_min(1e-8) * ref_norms
    return noise.to(dtype=reference.dtype)


def _prediction(scores: Sequence[float] | torch.Tensor) -> int:
    if isinstance(scores, torch.Tensor):
        values = [float(value) for value in scores.detach().cpu()]
    else:
        values = [float(value) for value in scores]
    return int(max(range(len(values)), key=lambda index: (values[index], -index)))


def _kl_to_full(condition_scores: Sequence[float], full_scores: Sequence[float]) -> float:
    condition = torch.tensor(list(condition_scores), dtype=torch.float64)
    full = torch.tensor(list(full_scores), dtype=torch.float64)
    full_probs = torch.softmax(full, dim=-1)
    log_condition = torch.log_softmax(condition, dim=-1)
    return float(torch.nn.functional.kl_div(log_condition, full_probs, reduction="sum").detach().cpu())


def _margin(scores: Sequence[float], answer_index: int) -> float:
    answer_index = int(answer_index)
    gold = float(scores[answer_index])
    others = [float(score) for index, score in enumerate(scores) if index != answer_index]
    return gold - max(others) if others else gold


def _paired_ci(
    *,
    selected: np.ndarray,
    baseline: np.ndarray,
    answers: np.ndarray,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    selected = np.asarray(selected, dtype=np.int64)
    baseline = np.asarray(baseline, dtype=np.int64)
    answers = np.asarray(answers, dtype=np.int64)
    delta = (selected == answers).astype(np.float64) - (baseline == answers).astype(np.float64)
    mean = float(delta.mean()) if delta.size else 0.0
    if delta.size <= 1 or samples <= 0:
        return {"mean_delta": mean, "ci95_low": mean, "ci95_high": mean, "samples": 0}
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    n = int(delta.size)
    for sample_index in range(int(samples)):
        indices = rng.integers(0, n, size=n)
        draws[sample_index] = float(delta[indices].mean())
    return {
        "mean_delta": mean,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "samples": int(samples),
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
    full_predictions = np.asarray(
        [int(row["prediction_index"]) for row in by_condition["full_prompt"]],
        dtype=np.int64,
    )
    answers = np.asarray(
        [int(row["answer_index"]) for row in by_condition["full_prompt"]],
        dtype=np.int64,
    )
    for condition, rows in by_condition.items():
        predictions = np.asarray([int(row["prediction_index"]) for row in rows], dtype=np.int64)
        correct = predictions == answers if len(predictions) == len(answers) else np.asarray([], dtype=bool)
        full_agreement = predictions == full_predictions if len(predictions) == len(full_predictions) else np.asarray([], dtype=bool)
        kl_values = [float(row["kl_to_full"]) for row in rows if math.isfinite(float(row["kl_to_full"]))]
        margins = [float(row["margin"]) for row in rows]
        metrics[condition] = {
            "n": int(len(rows)),
            "accuracy": float(correct.mean()) if correct.size else 0.0,
            "agreement_with_full_prompt": float(full_agreement.mean()) if full_agreement.size else 0.0,
            "mean_kl_to_full": float(statistics.fmean(kl_values)) if kl_values else 0.0,
            "median_kl_to_full": float(statistics.median(kl_values)) if kl_values else 0.0,
            "mean_margin": float(statistics.fmean(margins)) if margins else 0.0,
        }
        if condition != "full_prompt" and len(predictions) == len(answers):
            metrics[condition]["paired_vs_full_prompt_accuracy"] = _paired_ci(
                selected=predictions,
                baseline=full_predictions,
                answers=answers,
                seed=seed + len(metrics) * 997,
                samples=bootstrap_samples,
            )
    return metrics


def _optimize_prefix(
    *,
    target_model: Any,
    embed_tokens: Any,
    initial_prefix: torch.Tensor,
    anchor_ids: torch.Tensor,
    continuation_ids: Sequence[torch.Tensor],
    full_scores: torch.Tensor,
    steps: int,
    lr: float,
    top1_weight: float,
    norm_weight: float,
    embed_rms: float,
    length_normalize: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prefix = torch.nn.Parameter(initial_prefix.detach().clone())
    optimizer = torch.optim.AdamW([prefix], lr=float(lr))
    full_probs = torch.softmax(full_scores.detach().float(), dim=-1)
    full_top1 = torch.tensor([int(torch.argmax(full_scores.detach()).item())], dtype=torch.long, device=prefix.device)
    losses: list[float] = []
    kls: list[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        context = _prefix_context(prefix=prefix, anchor_ids=anchor_ids, embed_tokens=embed_tokens)
        soft_scores = _choice_scores_from_context(
            target_model=target_model,
            embed_tokens=embed_tokens,
            context_embeds=context,
            continuation_ids=continuation_ids,
            length_normalize=length_normalize,
        )
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(soft_scores.float(), dim=-1),
            full_probs,
            reduction="sum",
        )
        ce = torch.nn.functional.cross_entropy(soft_scores.unsqueeze(0), full_top1)
        token_rms = prefix.float().pow(2).mean(dim=1).sqrt()
        norm_loss = torch.mean((token_rms - float(embed_rms)) ** 2)
        loss = kl + float(top1_weight) * ce + float(norm_weight) * norm_loss
        losses.append(float(loss.detach().cpu()))
        kls.append(float(kl.detach().cpu()))
        loss.backward()
        optimizer.step()
    return prefix.detach(), {
        "loss_initial": float(losses[0]) if losses else 0.0,
        "loss_final": float(losses[-1]) if losses else 0.0,
        "kl_initial": float(kls[0]) if kls else 0.0,
        "kl_final": float(kls[-1]) if kls else 0.0,
        "steps": int(steps),
        "lr": float(lr),
        "top1_weight": float(top1_weight),
        "norm_weight": float(norm_weight),
    }


def _write_markdown(path: pathlib.Path | str, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    metrics = payload["metrics"]
    lines = [
        "# Target Self-Resonance HellaSwag Soft-Prefix Gate",
        "",
        f"- date: `{payload['date']}`",
        f"- artifact: `{payload['artifact_dir']}`",
        f"- target model: `{payload['target_model_path']}`",
        f"- rows: `{payload['row_count']}`",
        f"- prefix tokens: `{payload['prefix_len']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        "",
        "## Result",
        "",
        f"- full-prompt accuracy: `{metrics['full_prompt']['accuracy']:.6f}`",
        f"- optimized-prefix accuracy: `{metrics['optimized_soft_prefix']['accuracy']:.6f}`",
        f"- optimized-prefix agreement with full prompt: `{metrics['optimized_soft_prefix']['agreement_with_full_prompt']:.6f}`",
        f"- optimized-prefix mean KL to full prompt: `{metrics['optimized_soft_prefix']['mean_kl_to_full']:.6f}`",
        f"- chunk-mean prefix agreement: `{headline['chunk_mean_agreement']:.6f}`",
        f"- chunk-mean prefix mean KL: `{headline['chunk_mean_kl']:.6f}`",
        f"- optimized KL gain vs chunk-mean: `{headline['kl_gain_vs_chunk_mean']:.6f}`",
        f"- best destructive-control agreement: `{headline['best_destructive_agreement']:.6f}` (`{headline['best_destructive_by_agreement']}`)",
        f"- best destructive-control mean KL: `{headline['best_destructive_mean_kl']:.6f}` (`{headline['best_destructive_by_kl']}`)",
        f"- optimized KL gain vs best destructive: `{headline['kl_gain_vs_best_destructive']:.6f}`",
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Next Gate",
            "",
            payload["next_exact_gate"],
            "",
        ]
    )
    _resolve(path).parent.mkdir(parents=True, exist_ok=True)
    _resolve(path).write_text("\n".join(lines), encoding="utf-8")


def build_gate(
    *,
    output_dir: pathlib.Path,
    eval_path: pathlib.Path,
    target_model_path: str,
    row_start: int,
    row_limit: int,
    prefix_len: int,
    steps: int,
    lr: float,
    top1_weight: float,
    norm_weight: float,
    seed: int,
    device: str,
    dtype: str,
    max_length: int,
    anchor_text: str,
    continuation_mode: str,
    length_normalize: bool,
    min_agreement_gap: float,
    min_kl_gain_vs_chunk: float,
    min_kl_gain_vs_destructive: float,
    max_mean_kl: float,
    bootstrap_samples: int,
    local_files_only: bool,
    run_date: str,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = _resolve(eval_path)
    rows = arc_gate._load_rows(eval_path)
    selected_rows = rows[int(row_start) : int(row_start) + int(row_limit)]
    if not selected_rows:
        raise ValueError("selected no rows")
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
    embed_weight = embed_tokens.weight.detach().float()
    embed_rms = float(embed_weight.pow(2).mean(dim=1).sqrt().median().detach().cpu())
    embed_dim = int(embed_tokens.embedding_dim)
    anchor_ids = _encode_ids(tokenizer, anchor_text, device=resolved_device, add_special_tokens=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    prediction_rows: list[dict[str, Any]] = []
    row_summaries: list[dict[str, Any]] = []
    optimized_prefixes: list[torch.Tensor] = []
    optimized_scores_by_row: list[list[float]] = []

    for row_offset, row in enumerate(selected_rows):
        row_seed = int(seed) * 1009 + int(row_start) + row_offset
        prompt_ids = _encode_ids(
            tokenizer,
            _prompt(row),
            device=resolved_device,
            add_special_tokens=True,
        )[-int(max_length) :]
        choice_ids = [
            _encode_ids(
                tokenizer,
                _continuation(row, index, mode=continuation_mode),
                device=resolved_device,
                add_special_tokens=False,
            )
            for index, _ in enumerate(row.choices)
        ]
        with torch.no_grad():
            full_scores = _full_prompt_scores(
                target_model=model,
                embed_tokens=embed_tokens,
                prompt_ids=prompt_ids,
                continuation_ids=choice_ids,
                length_normalize=length_normalize,
            )
            chunk_prefix = _prompt_chunk_mean_prefix(
                prompt_ids=prompt_ids,
                embed_tokens=embed_tokens,
                prefix_len=prefix_len,
                embed_rms=embed_rms,
                device=resolved_device,
            )
        optimized_prefix, fit_log = _optimize_prefix(
            target_model=model,
            embed_tokens=embed_tokens,
            initial_prefix=chunk_prefix,
            anchor_ids=anchor_ids,
            continuation_ids=choice_ids,
            full_scores=full_scores,
            steps=steps,
            lr=lr,
            top1_weight=top1_weight,
            norm_weight=norm_weight,
            embed_rms=embed_rms,
            length_normalize=length_normalize,
        )
        optimized_prefixes.append(optimized_prefix.detach().cpu())

        with torch.no_grad():
            zero_prefix = torch.zeros_like(optimized_prefix)
            random_prefix = _random_same_norm_prefix(reference=optimized_prefix, seed=row_seed + 77)
            condition_score_tensors = {
                "full_prompt": full_scores,
                "chunk_mean_prefix": _choice_scores_from_context(
                    target_model=model,
                    embed_tokens=embed_tokens,
                    context_embeds=_prefix_context(
                        prefix=chunk_prefix,
                        anchor_ids=anchor_ids,
                        embed_tokens=embed_tokens,
                    ),
                    continuation_ids=choice_ids,
                    length_normalize=length_normalize,
                ),
                "optimized_soft_prefix": _choice_scores_from_context(
                    target_model=model,
                    embed_tokens=embed_tokens,
                    context_embeds=_prefix_context(
                        prefix=optimized_prefix,
                        anchor_ids=anchor_ids,
                        embed_tokens=embed_tokens,
                    ),
                    continuation_ids=choice_ids,
                    length_normalize=length_normalize,
                ),
                "zero_prefix": _choice_scores_from_context(
                    target_model=model,
                    embed_tokens=embed_tokens,
                    context_embeds=_prefix_context(
                        prefix=zero_prefix,
                        anchor_ids=anchor_ids,
                        embed_tokens=embed_tokens,
                    ),
                    continuation_ids=choice_ids,
                    length_normalize=length_normalize,
                ),
                "random_same_norm_prefix": _choice_scores_from_context(
                    target_model=model,
                    embed_tokens=embed_tokens,
                    context_embeds=_prefix_context(
                        prefix=random_prefix,
                        anchor_ids=anchor_ids,
                        embed_tokens=embed_tokens,
                    ),
                    continuation_ids=choice_ids,
                    length_normalize=length_normalize,
                ),
            }
        optimized_scores_by_row.append(
            [float(value) for value in condition_score_tensors["optimized_soft_prefix"].detach().cpu()]
        )
        full_score_list = [float(value) for value in full_scores.detach().cpu()]
        row_summaries.append(
            {
                "row_id": row.row_id,
                "content_id": row.content_id,
                "answer_index": int(row.answer_index),
                "answer_label": row.answer_label,
                "full_prompt_prediction": int(_prediction(full_scores)),
                "optimized_prediction": int(_prediction(condition_score_tensors["optimized_soft_prefix"])),
                "fit_loss_initial": float(fit_log["loss_initial"]),
                "fit_loss_final": float(fit_log["loss_final"]),
                "fit_kl_initial": float(fit_log["kl_initial"]),
                "fit_kl_final": float(fit_log["kl_final"]),
                "optimized_prefix_rms": float(optimized_prefix.float().pow(2).mean().sqrt().detach().cpu()),
                "chunk_prefix_rms": float(chunk_prefix.float().pow(2).mean().sqrt().detach().cpu()),
            }
        )
        for condition, score_tensor in condition_score_tensors.items():
            scores = [float(value) for value in score_tensor.detach().cpu()]
            pred = _prediction(scores)
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
                    "full_prompt_prediction_index": int(_prediction(full_score_list)),
                    "full_prompt_prediction_label": row.choice_labels[int(_prediction(full_score_list))],
                    "agrees_with_full_prompt": bool(pred == _prediction(full_score_list)),
                    "margin": float(_margin(scores, row.answer_index)),
                    "kl_to_full": float(_kl_to_full(scores, full_score_list)),
                    "scores": scores,
                }
            )

    learned_prefix_stack = torch.stack(optimized_prefixes, dim=0)
    for row_offset, row in enumerate(selected_rows):
        shuffled_prefix = learned_prefix_stack[(row_offset + 1) % len(selected_rows)].to(
            device=resolved_device,
            dtype=embed_tokens.weight.dtype,
        )
        prompt_ids = _encode_ids(
            tokenizer,
            _prompt(row),
            device=resolved_device,
            add_special_tokens=True,
        )[-int(max_length) :]
        choice_ids = [
            _encode_ids(
                tokenizer,
                _continuation(row, index, mode=continuation_mode),
                device=resolved_device,
                add_special_tokens=False,
            )
            for index, _ in enumerate(row.choices)
        ]
        with torch.no_grad():
            full_scores = _full_prompt_scores(
                target_model=model,
                embed_tokens=embed_tokens,
                prompt_ids=prompt_ids,
                continuation_ids=choice_ids,
                length_normalize=length_normalize,
            )
            shuffled_scores = _choice_scores_from_context(
                target_model=model,
                embed_tokens=embed_tokens,
                context_embeds=_prefix_context(
                    prefix=shuffled_prefix,
                    anchor_ids=anchor_ids,
                    embed_tokens=embed_tokens,
                ),
                continuation_ids=choice_ids,
                length_normalize=length_normalize,
            )
        full_score_list = [float(value) for value in full_scores.detach().cpu()]
        for condition, scores in {
            "shuffled_optimized_prefix": [float(value) for value in shuffled_scores.detach().cpu()],
            "candidate_derangement": list(np.roll(optimized_scores_by_row[row_offset], 1)),
        }.items():
            pred = _prediction(scores)
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
                    "full_prompt_prediction_index": int(_prediction(full_score_list)),
                    "full_prompt_prediction_label": row.choice_labels[int(_prediction(full_score_list))],
                    "agrees_with_full_prompt": bool(pred == _prediction(full_score_list)),
                    "margin": float(_margin(scores, row.answer_index)),
                    "kl_to_full": float(_kl_to_full(scores, full_score_list)),
                    "scores": [float(score) for score in scores],
                }
            )

    metrics = _condition_metrics(
        prediction_rows,
        seed=seed + 4242,
        bootstrap_samples=bootstrap_samples,
    )
    context_compression_baseline = "chunk_mean_prefix"
    destructive_controls = (
        "zero_prefix",
        "random_same_norm_prefix",
        "shuffled_optimized_prefix",
        "candidate_derangement",
    )
    best_destructive_by_agreement = max(
        destructive_controls,
        key=lambda condition: metrics[condition]["agreement_with_full_prompt"],
    )
    best_destructive_by_kl = min(
        destructive_controls,
        key=lambda condition: metrics[condition]["mean_kl_to_full"],
    )
    optimized = metrics["optimized_soft_prefix"]
    chunk = metrics[context_compression_baseline]
    best_destructive_agreement = float(
        metrics[best_destructive_by_agreement]["agreement_with_full_prompt"]
    )
    kl_gain_vs_chunk = float(chunk["mean_kl_to_full"] - optimized["mean_kl_to_full"])
    kl_gain_vs_best_destructive = float(
        metrics[best_destructive_by_kl]["mean_kl_to_full"] - optimized["mean_kl_to_full"]
    )
    pass_gate = bool(
        optimized["agreement_with_full_prompt"] >= best_destructive_agreement + float(min_agreement_gap)
        and optimized["agreement_with_full_prompt"] >= float(chunk["agreement_with_full_prompt"])
        and kl_gain_vs_chunk >= float(min_kl_gain_vs_chunk)
        and kl_gain_vs_best_destructive >= float(min_kl_gain_vs_destructive)
        and optimized["mean_kl_to_full"] <= float(max_mean_kl)
    )
    headline = {
        "optimized_agreement": float(optimized["agreement_with_full_prompt"]),
        "optimized_accuracy": float(optimized["accuracy"]),
        "optimized_mean_kl": float(optimized["mean_kl_to_full"]),
        "chunk_mean_agreement": float(chunk["agreement_with_full_prompt"]),
        "chunk_mean_kl": float(chunk["mean_kl_to_full"]),
        "kl_gain_vs_chunk_mean": kl_gain_vs_chunk,
        "best_destructive_by_agreement": best_destructive_by_agreement,
        "best_destructive_agreement": best_destructive_agreement,
        "best_destructive_by_kl": best_destructive_by_kl,
        "best_destructive_mean_kl": float(metrics[best_destructive_by_kl]["mean_kl_to_full"]),
        "kl_gain_vs_best_destructive": kl_gain_vs_best_destructive,
        "agreement_gap_vs_best_destructive": float(
            optimized["agreement_with_full_prompt"] - best_destructive_agreement
        ),
    }
    interpretation = (
        "The target-self branch stays alive if optimized soft prefixes preserve the frozen target "
        "model's full-context choice behavior more reliably than chunk-mean, zero, random, shuffled, "
        "and candidate-deranged controls. This does not yet prove a source-private encoder; it only "
        "tests whether the target has a compact continuous control surface worth learning into."
        if pass_gate
        else "This run does not promote target self-resonance as a positive method yet. Either the "
        "prefix budget/optimization is too weak, or full-context behavior is not reachable from this "
        "compact context-free soft prefix on the tested slice."
    )
    next_exact_gate = (
        "Train a generalizing text-to-prefix encoder on official train rows and evaluate held-out "
        "rows with the same destructive controls; then repeat with Phi as target if Mac runtime is tolerable."
        if pass_gate
        else "Run a bounded rescue with a larger prefix and logit+hidden matching, or kill this branch "
        "and move to a target-side query-resampler/ICAE-style encoder."
    )
    payload: dict[str, Any] = {
        "date": run_date,
        "artifact_dir": _display(output_dir),
        "pass_gate": pass_gate,
        "headline": headline,
        "metrics": metrics,
        "row_summaries": row_summaries,
        "row_count": int(len(selected_rows)),
        "row_start": int(row_start),
        "row_limit": int(row_limit),
        "eval_path": _display(eval_path),
        "eval_sha256": _sha256_file(eval_path),
        "target_model_path": str(target_model_path),
        "target_device": resolved_device,
        "dtype": dtype,
        "max_length": int(max_length),
        "prefix_len": int(prefix_len),
        "prefix_raw_bytes_fp16": int(prefix_len) * int(embed_dim) * 2,
        "prefix_raw_bytes_fp32": int(prefix_len) * int(embed_dim) * 4,
        "embed_dim": int(embed_dim),
        "embed_rms_median": float(embed_rms),
        "anchor_text": anchor_text,
        "anchor_token_count": int(anchor_ids.numel()),
        "continuation_mode": continuation_mode,
        "length_normalize": bool(length_normalize),
        "fit": {
            "steps": int(steps),
            "lr": float(lr),
            "top1_weight": float(top1_weight),
            "norm_weight": float(norm_weight),
        },
        "bootstrap_samples": int(bootstrap_samples),
        "pass_criteria": {
            "min_agreement_gap": float(min_agreement_gap),
            "min_kl_gain_vs_chunk": float(min_kl_gain_vs_chunk),
            "min_kl_gain_vs_destructive": float(min_kl_gain_vs_destructive),
            "max_mean_kl": float(max_mean_kl),
        },
        "interpretation": interpretation,
        "next_exact_gate": next_exact_gate,
        "runtime_s": float(time.perf_counter() - start_time),
        "peak_rss_mib": float(_peak_rss_mib()),
        "claim_boundary": (
            "Per-example optimized target soft prefixes are an oracle capacity probe. They are not "
            "a learned cross-model source-private channel until a held-out encoder emits the prefix "
            "from allowed source observations and beats the same controls."
        ),
    }
    json_path = output_dir / "target_self_resonance_hellaswag_soft_prefix_gate.json"
    md_path = output_dir / "target_self_resonance_hellaswag_soft_prefix_gate.md"
    predictions_path = output_dir / "predictions.jsonl"
    row_summary_path = output_dir / "row_summaries.csv"
    _write_json(json_path, payload)
    _write_markdown(md_path, payload)
    _write_jsonl(predictions_path, prediction_rows)
    _write_csv(row_summary_path, row_summaries)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            "target_self_resonance_hellaswag_soft_prefix_gate.json": _sha256_file(json_path),
            "target_self_resonance_hellaswag_soft_prefix_gate.md": _sha256_file(md_path),
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
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=16)
    parser.add_argument("--prefix-len", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--top1-weight", type=float, default=0.0)
    parser.add_argument("--norm-weight", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--anchor-text", default="Continuation:")
    parser.add_argument("--continuation-mode", choices=("choice", "label_and_choice", "label"), default="choice")
    parser.add_argument("--length-normalize", choices=("true", "false"), default="true")
    parser.add_argument("--min-agreement-gap", type=float, default=0.0)
    parser.add_argument("--min-kl-gain-vs-chunk", type=float, default=0.0)
    parser.add_argument("--min-kl-gain-vs-destructive", type=float, default=0.01)
    parser.add_argument("--max-mean-kl", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--run-date", default=str(dt.date.today()))
    args = parser.parse_args()
    payload = build_gate(
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        target_model_path=args.target_model_path,
        row_start=args.row_start,
        row_limit=args.row_limit,
        prefix_len=args.prefix_len,
        steps=args.steps,
        lr=args.lr,
        top1_weight=args.top1_weight,
        norm_weight=args.norm_weight,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        anchor_text=args.anchor_text,
        continuation_mode=args.continuation_mode,
        length_normalize=args.length_normalize == "true",
        min_agreement_gap=args.min_agreement_gap,
        min_kl_gain_vs_chunk=args.min_kl_gain_vs_chunk,
        min_kl_gain_vs_destructive=args.min_kl_gain_vs_destructive,
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
