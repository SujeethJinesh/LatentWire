from __future__ import annotations

"""Run a Mac-local target-loss soft-prefix preflight for ARC/OpenBookQA.

This is an implementation gate, not a paper-positive result.  It trains only a
small connector that maps answer-key-forbidden source summaries into target
input-embedding prefix tokens, then scores multiple-choice continuations with a
frozen target LM.  The readout is whether the matched source prefix beats
target-only/static/source-destroying controls on a tiny held-out slice.
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
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate  # noqa: E402


DEFAULT_OUTPUT = pathlib.Path("results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_smoke")
DEFAULT_ARC_VALIDATION = pathlib.Path(
    "results/source_private_arc_challenge_bridge_contract_20260501/official_splits/arc_challenge_validation.jsonl"
)
DEFAULT_ARC_SOURCE_CACHE = pathlib.Path(
    "results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/source_prediction_cache.jsonl"
)
DEFAULT_QWEN_SOURCE = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
    "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
DEFAULT_QWEN_TARGET = (
    "/Users/sujeethjinesh/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)

MATCHED_CONDITION = "matched_soft_prefix"
CONTROL_CONDITIONS = (
    "target_only",
    "target_cache_only_prefix",
    "slots_only_prefix",
    "zero_source",
    "shuffled_source",
    "same_norm_noise",
    "train_mean_source",
    "label_shuffled",
    "candidate_derangement",
    "same_byte_visible_text",
    "source_label_copy_audit_upper_bound",
)
PASS_CONTROL_CONDITIONS = tuple(
    condition for condition in CONTROL_CONDITIONS if condition != "source_label_copy_audit_upper_bound"
)
REPORT_CONDITIONS = (MATCHED_CONDITION, *CONTROL_CONDITIONS)


@dataclass(frozen=True)
class SoftPrefixConfig:
    prefix_len: int = 4
    hidden_dim: int = 32
    epochs: int = 2
    lr: float = 3e-3
    weight_decay: float = 1e-3
    seed: int = 17
    matched_use_target: bool = False
    length_normalize: bool = True


class SourceSoftPrefixConnector(torch.nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        target_dim: int,
        target_embed_dim: int,
        hidden_dim: int,
        prefix_len: int,
        use_source: bool,
        use_target: bool,
    ) -> None:
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.target_embed_dim = int(target_embed_dim)
        self.use_source = bool(use_source)
        self.use_target = bool(use_target)
        input_dim = (int(source_dim) if self.use_source else 0) + (
            int(target_dim) if self.use_target else 0
        )
        if input_dim == 0:
            self.slots = torch.nn.Parameter(torch.randn(prefix_len, target_embed_dim) * 0.02)
            self.net = None
        else:
            self.slots = None
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, int(hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.Linear(int(hidden_dim), int(prefix_len) * int(target_embed_dim)),
            )

    def forward(self, source_summary: torch.Tensor, target_summary: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return self.slots
        parts: list[torch.Tensor] = []
        if self.use_source:
            parts.append(source_summary)
        if self.use_target:
            parts.append(target_summary)
        return self.net(torch.cat(parts, dim=-1)).view(self.prefix_len, self.target_embed_dim)


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


def _peak_rss_mib() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def _torch_dtype(dtype: str) -> Any:
    return arc_gate._torch_dtype(dtype)


def _read_source_cache(path: pathlib.Path) -> dict[str, int]:
    predictions: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = set(row.get("forbidden_source_fields", ()))
            if not set(arc_gate.FORBIDDEN_SOURCE_KEYS) <= forbidden:
                raise ValueError(f"source cache row {row.get('row_id')} is missing forbidden fields")
            predictions[str(row["content_id"])] = int(row["source_selected_index"])
    if not predictions:
        raise ValueError(f"{path} contained no source predictions")
    return predictions


def _select_rows_with_cache(
    rows: Sequence[arc_gate.ArcRow],
    source_cache: dict[str, int],
    *,
    row_limit: int,
) -> tuple[list[arc_gate.ArcRow], list[int]]:
    selected: list[arc_gate.ArcRow] = []
    predictions: list[int] = []
    for row in rows:
        if row.content_id not in source_cache:
            continue
        prediction = int(source_cache[row.content_id])
        if 0 <= prediction < len(row.choices):
            selected.append(row)
            predictions.append(prediction)
        if len(selected) >= row_limit:
            break
    if len(selected) < 2:
        raise ValueError("need at least two rows with source-cache predictions")
    return selected, predictions


def _mcq_prompt(row: arc_gate.ArcRow) -> str:
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(row.choice_labels, row.choices, strict=True)
    )
    return (
        "Answer the science multiple-choice question. Use only the listed choices.\n"
        f"Question: {row.question}\n"
        f"Choices:\n{choices}\n"
        "Best answer:"
    )


def _source_prompt(row: arc_gate.ArcRow) -> str:
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(row.choice_labels, row.choices, strict=True)
    )
    return (
        "Read the science question and choices. Do not reveal the answer label.\n"
        f"Question: {row.question}\nChoices:\n{choices}\n"
        "Useful evidence:"
    )


def _row_public_text(row: arc_gate.ArcRow) -> str:
    return _source_prompt(row)


def _source_choice_texts(rows: list[arc_gate.ArcRow]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        prompt = _source_prompt(row)
        for label, choice in zip(row.choice_labels, row.choices, strict=True):
            texts.append(f"{prompt}\nCandidate under consideration: {label}. {choice}")
    return texts


def _continuation_text(row: arc_gate.ArcRow, choice_index: int, *, mode: str) -> str:
    if mode == "label":
        return f" {row.choice_labels[choice_index]}"
    if mode == "label_and_choice":
        return f" {row.choice_labels[choice_index]}. {row.choices[choice_index]}"
    if mode == "choice":
        return f" {row.choices[choice_index]}"
    raise ValueError(f"unknown continuation mode {mode!r}")


def _encode_ids(tokenizer: Any, text: str, *, device: str, add_special_tokens: bool) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
    ids = encoded.input_ids[0].to(device)
    if ids.numel() == 0:
        raise ValueError(f"zero-token text: {text!r}")
    return ids


def _standardize(matrix: torch.Tensor, train_indices: Sequence[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    train = matrix[list(train_indices)]
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (matrix - mean) / std, {
        "mean_l2": float(mean.norm().detach().cpu()),
        "std_min": float(std.min().detach().cpu()),
        "std_max": float(std.max().detach().cpu()),
    }


def _row_indices(
    *,
    row_count: int,
    fit_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be between 0 and 1")
    indices = list(range(row_count))
    random.Random(seed).shuffle(indices)
    fit_count = max(1, min(row_count - 1, int(round(row_count * fit_fraction))))
    fit = sorted(indices[:fit_count])
    eval_indices = sorted(indices[fit_count:])
    return fit, eval_indices


def _selected_choice_features(
    rows: list[arc_gate.ArcRow],
    source_predictions: list[int],
    *,
    source_feature_mode: str,
    feature_dim: int,
    source_model: str,
    source_device: str,
    source_dtype: str,
    source_max_length: int,
    source_hidden_layer: int,
    local_files_only: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if source_feature_mode == "hashed_selected":
        flat = arc_gate._features(
            arc_gate._choice_pair_texts(rows),
            dim=feature_dim,
            feature_mode="hashed",
            feature_model="",
            feature_device="auto",
            feature_dtype="float32",
            feature_max_length=source_max_length,
            local_files_only=True,
        )
        metadata = {"kind": "hashed_selected_choice", "feature_dim": int(feature_dim)}
    elif source_feature_mode == "hf_selected_hidden":
        flat, metadata = _hf_choice_hidden_features(
            rows,
            model_path=source_model,
            device=source_device,
            dtype=source_dtype,
            max_length=source_max_length,
            local_files_only=local_files_only,
            hidden_layer=source_hidden_layer,
        )
    else:
        raise ValueError(f"unknown source_feature_mode {source_feature_mode!r}")

    chosen: list[np.ndarray] = []
    offset = 0
    for row, selected_index in zip(rows, source_predictions, strict=True):
        chosen.append(flat[offset + int(selected_index)])
        offset += len(row.choices)
    return torch.tensor(np.asarray(chosen, dtype=np.float32)), metadata


def _hf_choice_hidden_features(
    rows: list[arc_gate.ArcRow],
    *,
    model_path: str,
    device: str,
    dtype: str,
    max_length: int,
    local_files_only: bool,
    hidden_layer: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = "cpu" if device == "auto_cpu" else arc_gate.syn._resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=_torch_dtype(dtype),
    ).to(resolved_device)
    model.eval()

    features: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for text in _source_choice_texts(rows):
            encoded = tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[hidden_layer][0]
            mask = encoded["attention_mask"][0].bool()
            values = hidden[mask]
            feature = values.mean(dim=0).detach().cpu().numpy().astype(np.float64)
            norm = np.linalg.norm(feature)
            features.append(feature / max(norm, 1e-12))
    return np.asarray(features, dtype=np.float64), {
        "kind": "answer_key_forbidden_hf_choice_hidden",
        "model_path": model_path,
        "device": resolved_device,
        "dtype": dtype,
        "max_length": int(max_length),
        "hidden_layer": int(hidden_layer),
        "latency_s": float(time.perf_counter() - start),
    }


def _target_public_features(rows: list[arc_gate.ArcRow], *, feature_dim: int) -> tuple[torch.Tensor, dict[str, Any]]:
    features = arc_gate._hashed_features([_row_public_text(row) for row in rows], dim=feature_dim)
    return torch.tensor(features.astype(np.float32)), {
        "kind": "hashed_public_question_choices",
        "feature_dim": int(feature_dim),
    }


def _continuation_logprob(
    *,
    target_model: Any,
    embed_tokens: Any,
    prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    length_normalize: bool,
) -> torch.Tensor:
    device = prefix.device if prefix.numel() else prompt_ids.device
    prompt_embeds = embed_tokens(prompt_ids.to(device)).detach()
    continuation_embeds = embed_tokens(continuation_ids.to(device)).detach()
    prefix = prefix.to(device=device, dtype=prompt_embeds.dtype)
    if continuation_embeds.shape[0] > 1:
        inputs = torch.cat([prefix, prompt_embeds, continuation_embeds[:-1]], dim=0)
    else:
        inputs = torch.cat([prefix, prompt_embeds], dim=0)
    attention_mask = torch.ones((1, inputs.shape[0]), dtype=torch.long, device=device)
    out = target_model(inputs_embeds=inputs.unsqueeze(0), attention_mask=attention_mask, use_cache=False)
    logits = out.logits[0]
    start = int(prefix.shape[0] + prompt_embeds.shape[0] - 1)
    token_logits = logits[start : start + continuation_ids.shape[0]]
    logprobs = torch.log_softmax(token_logits.float(), dim=-1)
    score = logprobs.gather(1, continuation_ids.to(device)[:, None]).sum()
    if length_normalize:
        return score / max(int(continuation_ids.shape[0]), 1)
    return score


def _choice_scores(
    *,
    target_model: Any,
    embed_tokens: Any,
    prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    continuation_ids: Sequence[torch.Tensor],
    length_normalize: bool,
) -> torch.Tensor:
    return torch.stack(
        [
            _continuation_logprob(
                target_model=target_model,
                embed_tokens=embed_tokens,
                prefix=prefix,
                prompt_ids=prompt_ids,
                continuation_ids=ids,
                length_normalize=length_normalize,
            )
            for ids in continuation_ids
        ]
    )


def _fit_connector(
    *,
    connector: SourceSoftPrefixConnector,
    target_model: Any,
    embed_tokens: Any,
    source_summary: torch.Tensor,
    target_summary: torch.Tensor,
    prompt_ids: Sequence[torch.Tensor],
    continuation_ids: Sequence[Sequence[torch.Tensor]],
    answer_indices: Sequence[int],
    fit_indices: Sequence[int],
    config: SoftPrefixConfig,
    device: str,
    label_shuffle: bool,
) -> dict[str, float]:
    connector.to(device)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    fit_indices = list(fit_indices)
    if label_shuffle:
        shifted_answers = {
            idx: int(answer_indices[fit_indices[(pos + 1) % len(fit_indices)]])
            for pos, idx in enumerate(fit_indices)
        }
    else:
        shifted_answers = {idx: int(answer_indices[idx]) for idx in fit_indices}
    losses: list[float] = []
    for _ in range(config.epochs):
        total = torch.zeros((), device=device)
        for idx in fit_indices:
            prefix = connector(source_summary[idx].to(device), target_summary[idx].to(device))
            scores = _choice_scores(
                target_model=target_model,
                embed_tokens=embed_tokens,
                prefix=prefix,
                prompt_ids=prompt_ids[idx],
                continuation_ids=continuation_ids[idx],
                length_normalize=config.length_normalize,
            )
            label = torch.tensor([shifted_answers[idx]], dtype=torch.long, device=device)
            total = total + torch.nn.functional.cross_entropy(scores.unsqueeze(0), label)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    connector.eval()
    return {
        "loss_initial": float(losses[0]) if losses else 0.0,
        "loss_final": float(losses[-1]) if losses else 0.0,
    }


@torch.no_grad()
def _score_connector_condition(
    *,
    connector: SourceSoftPrefixConnector | None,
    target_model: Any,
    embed_tokens: Any,
    source_summary: torch.Tensor,
    target_summary: torch.Tensor,
    prompt_ids: torch.Tensor,
    continuation_ids: Sequence[torch.Tensor],
    length_normalize: bool,
    device: str,
) -> list[float]:
    if connector is None:
        embed_dim = int(embed_tokens.embedding_dim)
        prefix = torch.empty((0, embed_dim), dtype=embed_tokens.weight.dtype, device=device)
    else:
        prefix = connector(source_summary.to(device), target_summary.to(device))
    scores = _choice_scores(
        target_model=target_model,
        embed_tokens=embed_tokens,
        prefix=prefix,
        prompt_ids=prompt_ids,
        continuation_ids=continuation_ids,
        length_normalize=length_normalize,
    )
    return [float(value) for value in scores.detach().cpu()]


def _margin(scores: Sequence[float], answer_index: int) -> float:
    gold = float(scores[answer_index])
    distractors = [float(score) for index, score in enumerate(scores) if index != answer_index]
    return gold - max(distractors) if distractors else gold


def _prediction(scores: Sequence[float]) -> int:
    return int(max(range(len(scores)), key=lambda index: (float(scores[index]), -index)))


def _paired_bootstrap(deltas: list[float], *, seed: int, samples: int) -> dict[str, float]:
    if not deltas:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples)]
    return {
        "mean": float(statistics.fmean(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
    }


def _summarize_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "mean_margin": 0.0, "correct": 0}
    correct = sum(1 for row in rows if row["correct"])
    return {
        "n": len(rows),
        "correct": int(correct),
        "accuracy": float(correct / len(rows)),
        "mean_margin": float(statistics.fmean(float(row["margin"]) for row in rows)),
    }


def _condition_metrics(rows: list[dict[str, Any]], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    metrics = {
        condition: _summarize_condition([row for row in rows if row["condition"] == condition])
        for condition in REPORT_CONDITIONS
    }
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(str(row["content_id"]), {})[str(row["condition"])] = row
    for condition in CONTROL_CONDITIONS:
        deltas = [
            float(group[MATCHED_CONDITION]["correct"]) - float(group[condition]["correct"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and condition in group
        ]
        margin_deltas = [
            float(group[MATCHED_CONDITION]["margin"]) - float(group[condition]["margin"])
            for group in by_id.values()
            if MATCHED_CONDITION in group and condition in group
        ]
        metrics[MATCHED_CONDITION][f"paired_accuracy_vs_{condition}"] = _paired_bootstrap(
            deltas,
            seed=seed + len(condition),
            samples=bootstrap_samples,
        )
        metrics[MATCHED_CONDITION][f"mean_margin_delta_vs_{condition}"] = (
            float(statistics.fmean(margin_deltas)) if margin_deltas else 0.0
        )
    return metrics


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "row_id",
        "content_id",
        "condition",
        "answer_index",
        "prediction_index",
        "correct",
        "margin",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    headline = payload["headline"]
    lines = [
        "# ARC/OpenBookQA Soft-Prefix Preflight",
        "",
        f"- date: `{payload['date']}`",
        f"- benchmark: `{payload['benchmark']}`",
        f"- pass gate: `{payload['pass_gate']}`",
        f"- fit/eval rows: `{payload['fit_rows']}` / `{payload['eval_rows']}`",
        f"- matched accuracy: `{headline['matched_accuracy']:.3f}`",
        f"- best control by accuracy: `{headline['best_control_by_accuracy']}`",
        f"- best control accuracy: `{headline['best_control_accuracy']:.3f}`",
        f"- matched margin: `{headline['matched_mean_margin']:.6f}`",
        f"- best control margin: `{headline['best_control_mean_margin']:.6f}`",
        f"- matched minus best-control margin: `{headline['matched_minus_best_control_margin']:.6f}`",
        "",
        "## Conditions",
        "",
        "| Condition | Accuracy | Correct / N | Mean Margin |",
        "|---|---:|---:|---:|",
    ]
    for condition, metrics in payload["condition_metrics"].items():
        lines.append(
            f"| `{condition}` | {metrics['accuracy']:.3f} | {metrics['correct']} / {metrics['n']} | "
            f"{metrics['mean_margin']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "Lay explanation: the experiment trains a tiny translator that turns an answer-key-forbidden "
            "source-model summary into soft tokens prepended to the target model. The controls ask whether "
            "the soft tokens are really using the source row, or whether a static/target-only prefix can do "
            "the same thing.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_preflight(
    *,
    output_dir: pathlib.Path,
    eval_path: pathlib.Path,
    source_cache_path: pathlib.Path,
    benchmark: str,
    row_limit: int,
    fit_fraction: float,
    source_feature_mode: str,
    source_feature_dim: int,
    target_feature_dim: int,
    source_model: str,
    target_model_path: str,
    source_device: str,
    target_device: str,
    train_device: str | None,
    target_attn_implementation: str | None,
    dtype: str,
    source_max_length: int,
    target_max_length: int,
    source_hidden_layer: int,
    local_files_only: bool,
    prefix_len: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    bootstrap_samples: int,
    continuation_mode: str,
    matched_use_target: bool,
    length_normalize: bool,
    same_byte_budget: int,
    min_accuracy_gap: float,
    min_margin_gap: float,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    output_dir = _resolve(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_all = arc_gate._load_rows(_resolve(eval_path))
    source_cache = _read_source_cache(_resolve(source_cache_path))
    rows, source_predictions = _select_rows_with_cache(rows_all, source_cache, row_limit=row_limit)
    fit_indices, eval_indices = _row_indices(row_count=len(rows), fit_fraction=fit_fraction, seed=seed)

    source_summary, source_meta = _selected_choice_features(
        rows,
        source_predictions,
        source_feature_mode=source_feature_mode,
        feature_dim=source_feature_dim,
        source_model=source_model,
        source_device=source_device,
        source_dtype=dtype,
        source_max_length=source_max_length,
        source_hidden_layer=source_hidden_layer,
        local_files_only=local_files_only,
    )
    target_summary, target_meta = _target_public_features(rows, feature_dim=target_feature_dim)
    source_summary, source_standardizer = _standardize(source_summary, fit_indices)
    target_summary, target_standardizer = _standardize(target_summary, fit_indices)

    resolved_target_device = "cpu" if target_device == "auto_cpu" else arc_gate.syn._resolve_torch_device(target_device)
    resolved_train_device = (
        resolved_target_device
        if train_device is None
        else ("cpu" if train_device == "auto_cpu" else arc_gate.syn._resolve_torch_device(train_device))
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "torch_dtype": _torch_dtype(dtype),
    }
    if target_attn_implementation and target_attn_implementation != "auto":
        model_kwargs["attn_implementation"] = target_attn_implementation
    model = AutoModelForCausalLM.from_pretrained(target_model_path, **model_kwargs).to(resolved_target_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(resolved_train_device)
    embed_tokens = model.get_input_embeddings()

    prompt_ids = [
        _encode_ids(tokenizer, _mcq_prompt(row), device=resolved_train_device, add_special_tokens=True)[-target_max_length:]
        for row in rows
    ]
    choice_ids = [
        [
            _encode_ids(
                tokenizer,
                _continuation_text(row, index, mode=continuation_mode),
                device=resolved_train_device,
                add_special_tokens=False,
            )
            for index, _ in enumerate(row.choices)
        ]
        for row in rows
    ]
    answer_indices = [int(row.answer_index) for row in rows]

    config = SoftPrefixConfig(
        prefix_len=prefix_len,
        hidden_dim=hidden_dim,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        matched_use_target=matched_use_target,
        length_normalize=length_normalize,
    )
    torch.manual_seed(seed)
    source_summary = source_summary.to(resolved_train_device)
    target_summary = target_summary.to(resolved_train_device)
    source_dim = int(source_summary.shape[-1])
    target_dim = int(target_summary.shape[-1])
    embed_dim = int(embed_tokens.embedding_dim)

    connectors = {
        MATCHED_CONDITION: SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            use_source=True,
            use_target=matched_use_target,
        ),
        "target_cache_only_prefix": SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            use_source=False,
            use_target=True,
        ),
        "slots_only_prefix": SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            use_source=False,
            use_target=False,
        ),
        "label_shuffled": SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            prefix_len=prefix_len,
            use_source=True,
            use_target=matched_use_target,
        ),
    }
    fit_logs: dict[str, Any] = {}
    for name, connector in connectors.items():
        fit_logs[name] = _fit_connector(
            connector=connector,
            target_model=model,
            embed_tokens=embed_tokens,
            source_summary=source_summary,
            target_summary=target_summary,
            prompt_ids=prompt_ids,
            continuation_ids=choice_ids,
            answer_indices=answer_indices,
            fit_indices=fit_indices,
            config=config,
            device=resolved_train_device,
            label_shuffle=name == "label_shuffled",
        )

    train_mean_source = source_summary[fit_indices].mean(dim=0)
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for eval_position, idx in enumerate(eval_indices):
            row = rows[idx]
            shuffled_idx = eval_indices[(eval_position + 1) % len(eval_indices)] if len(eval_indices) > 1 else idx
            generator = torch.Generator(device="cpu").manual_seed(seed * 1009 + idx)
            noise_cpu = torch.randn(tuple(source_summary[idx].shape), generator=generator)
            noise = noise_cpu.to(resolved_train_device)
            noise = noise / noise.norm().clamp_min(1e-6) * source_summary[idx].norm().clamp_min(1e-6)
            source_variants = {
                MATCHED_CONDITION: source_summary[idx],
                "zero_source": torch.zeros_like(source_summary[idx]),
                "shuffled_source": source_summary[shuffled_idx],
                "same_norm_noise": noise,
                "train_mean_source": train_mean_source,
                "target_cache_only_prefix": source_summary[idx],
                "slots_only_prefix": source_summary[idx],
                "label_shuffled": source_summary[idx],
            }
            condition_scores: dict[str, list[float]] = {}
            condition_scores["target_only"] = _score_connector_condition(
                connector=None,
                target_model=model,
                embed_tokens=embed_tokens,
                source_summary=source_summary[idx],
                target_summary=target_summary[idx],
                prompt_ids=prompt_ids[idx],
                continuation_ids=choice_ids[idx],
                length_normalize=length_normalize,
                device=resolved_train_device,
            )
            for condition in (
                MATCHED_CONDITION,
                "zero_source",
                "shuffled_source",
                "same_norm_noise",
                "train_mean_source",
            ):
                condition_scores[condition] = _score_connector_condition(
                    connector=connectors[MATCHED_CONDITION],
                    target_model=model,
                    embed_tokens=embed_tokens,
                    source_summary=source_variants[condition],
                    target_summary=target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    continuation_ids=choice_ids[idx],
                    length_normalize=length_normalize,
                    device=resolved_train_device,
                )
            for condition in ("target_cache_only_prefix", "slots_only_prefix", "label_shuffled"):
                condition_scores[condition] = _score_connector_condition(
                    connector=connectors[condition],
                    target_model=model,
                    embed_tokens=embed_tokens,
                    source_summary=source_variants[condition],
                    target_summary=target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    continuation_ids=choice_ids[idx],
                    length_normalize=length_normalize,
                    device=resolved_train_device,
                )
            condition_scores["candidate_derangement"] = list(np.roll(condition_scores[MATCHED_CONDITION], 1))
            hint = row.choices[source_predictions[idx]].encode("utf-8")[:same_byte_budget].decode(
                "utf-8", errors="ignore"
            )
            hint_prompt = _mcq_prompt(row) + f"\nVisible same-byte hint: {hint}\nBest answer:"
            hint_prompt_ids = _encode_ids(
                tokenizer,
                hint_prompt,
                device=resolved_train_device,
                add_special_tokens=True,
            )[-target_max_length:]
            condition_scores["same_byte_visible_text"] = _score_connector_condition(
                connector=None,
                target_model=model,
                embed_tokens=embed_tokens,
                source_summary=source_summary[idx],
                target_summary=target_summary[idx],
                prompt_ids=hint_prompt_ids,
                continuation_ids=choice_ids[idx],
                length_normalize=length_normalize,
                device=resolved_train_device,
            )
            audit_scores = [-1.0e9 for _ in row.choices]
            audit_scores[source_predictions[idx]] = 0.0
            condition_scores["source_label_copy_audit_upper_bound"] = audit_scores
            for condition in REPORT_CONDITIONS:
                scores = condition_scores[condition]
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
                        "margin": float(_margin(scores, row.answer_index)),
                        "scores": [float(score) for score in scores],
                        "source_selected_index": int(source_predictions[idx]),
                        "source_selected_label": row.choice_labels[int(source_predictions[idx])],
                    }
                )

    metrics = _condition_metrics(prediction_rows, seed=seed + 404, bootstrap_samples=bootstrap_samples)
    matched = metrics[MATCHED_CONDITION]
    best_control_by_accuracy = max(PASS_CONTROL_CONDITIONS, key=lambda name: metrics[name]["accuracy"])
    best_control_by_margin = max(PASS_CONTROL_CONDITIONS, key=lambda name: metrics[name]["mean_margin"])
    headline = {
        "matched_accuracy": matched["accuracy"],
        "matched_mean_margin": matched["mean_margin"],
        "best_control_by_accuracy": best_control_by_accuracy,
        "best_control_accuracy": metrics[best_control_by_accuracy]["accuracy"],
        "best_control_by_margin": best_control_by_margin,
        "best_control_mean_margin": metrics[best_control_by_margin]["mean_margin"],
        "matched_minus_best_control_accuracy": matched["accuracy"] - metrics[best_control_by_accuracy]["accuracy"],
        "matched_minus_best_control_margin": matched["mean_margin"] - metrics[best_control_by_margin]["mean_margin"],
    }
    pass_gate = bool(
        headline["matched_minus_best_control_accuracy"] >= min_accuracy_gap
        and headline["matched_minus_best_control_margin"] >= min_margin_gap
        and matched[f"paired_accuracy_vs_{best_control_by_accuracy}"]["ci95_low"] > 0.0
    )
    interpretation = (
        "This preflight passes only if the matched soft-prefix uses source information that the "
        "target-only/static/shuffled/noise controls cannot reproduce. A failure is not a final "
        "scientific negative; it either kills this exact tiny Mac-local setup or exposes a target-cache leak."
    )
    payload = {
        "gate": "source_private_arc_openbookqa_soft_prefix_preflight",
        "date": "2026-05-02",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "benchmark": benchmark,
        "pass_gate": pass_gate,
        "implementation_gate_only": True,
        "fit_rows": len(fit_indices),
        "eval_rows": len(eval_indices),
        "row_limit": len(rows),
        "fit_indices": fit_indices,
        "eval_indices": eval_indices,
        "config": {
            "source_feature_mode": source_feature_mode,
            "source_feature_dim": source_feature_dim,
            "target_feature_dim": target_feature_dim,
            "prefix_len": prefix_len,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "matched_use_target": matched_use_target,
            "length_normalize": length_normalize,
            "continuation_mode": continuation_mode,
            "same_byte_budget": same_byte_budget,
            "source_model": source_model,
            "target_model": target_model_path,
            "source_device": source_device,
            "target_device": target_device,
            "train_device": resolved_train_device,
            "target_attn_implementation": target_attn_implementation or "auto",
            "dtype": dtype,
        },
        "feature_metadata": {
            "source": source_meta,
            "target": target_meta,
            "source_standardizer": source_standardizer,
            "target_standardizer": target_standardizer,
        },
        "fit_logs": fit_logs,
        "headline": headline,
        "pass_control_conditions": list(PASS_CONTROL_CONDITIONS),
        "audit_only_conditions": ["source_label_copy_audit_upper_bound"],
        "condition_metrics": metrics,
        "interpretation": interpretation,
        "inputs": {
            "eval_path": _display(eval_path),
            "source_cache_path": _display(source_cache_path),
        },
        "runtime": {
            "latency_s": float(time.perf_counter() - total_start),
            "peak_rss_mib": _peak_rss_mib(),
        },
    }
    json_path = output_dir / "arc_openbookqa_soft_prefix_preflight.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(output_dir / "prediction_audit.jsonl", prediction_rows)
    _write_csv(output_dir / "prediction_audit.csv", prediction_rows)
    _write_markdown(output_dir / "arc_openbookqa_soft_prefix_preflight.md", payload)
    manifest = {
        "gate": payload["gate"],
        "pass_gate": payload["pass_gate"],
        "created_utc": payload["created_utc"],
        "files": [
            {
                "path": _display(path),
                "sha256": _sha256_file(path),
                "bytes": _resolve(path).stat().st_size,
            }
            for path in (
                json_path,
                output_dir / "prediction_audit.jsonl",
                output_dir / "prediction_audit.csv",
                output_dir / "arc_openbookqa_soft_prefix_preflight.md",
            )
        ],
        "inputs": payload["inputs"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", default="ARC-Challenge")
    parser.add_argument("--eval-path", type=pathlib.Path, default=DEFAULT_ARC_VALIDATION)
    parser.add_argument("--source-cache-path", type=pathlib.Path, default=DEFAULT_ARC_SOURCE_CACHE)
    parser.add_argument("--row-limit", type=int, default=8)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--source-feature-mode", choices=("hashed_selected", "hf_selected_hidden"), default="hf_selected_hidden")
    parser.add_argument("--source-feature-dim", type=int, default=128)
    parser.add_argument("--target-feature-dim", type=int, default=64)
    parser.add_argument("--source-model", default=DEFAULT_QWEN_SOURCE)
    parser.add_argument("--target-model", default=DEFAULT_QWEN_TARGET)
    parser.add_argument("--source-device", default="auto_cpu")
    parser.add_argument("--target-device", default="mps")
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--target-attn-implementation", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--source-max-length", type=int, default=192)
    parser.add_argument("--target-max-length", type=int, default=256)
    parser.add_argument("--source-hidden-layer", type=int, default=-1)
    parser.add_argument("--local-files-only", choices=("true", "false"), default="true")
    parser.add_argument("--prefix-len", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--continuation-mode", choices=("label", "label_and_choice", "choice"), default="label")
    parser.add_argument("--matched-use-target", choices=("true", "false"), default="false")
    parser.add_argument("--length-normalize", choices=("true", "false"), default="true")
    parser.add_argument("--same-byte-budget", type=int, default=12)
    parser.add_argument("--min-accuracy-gap", type=float, default=0.0)
    parser.add_argument("--min-margin-gap", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_preflight(
        output_dir=args.output_dir,
        eval_path=args.eval_path,
        source_cache_path=args.source_cache_path,
        benchmark=str(args.benchmark),
        row_limit=int(args.row_limit),
        fit_fraction=float(args.fit_fraction),
        source_feature_mode=str(args.source_feature_mode),
        source_feature_dim=int(args.source_feature_dim),
        target_feature_dim=int(args.target_feature_dim),
        source_model=str(args.source_model),
        target_model_path=str(args.target_model),
        source_device=str(args.source_device),
        target_device=str(args.target_device),
        train_device=None if args.train_device is None else str(args.train_device),
        target_attn_implementation=str(args.target_attn_implementation),
        dtype=str(args.dtype),
        source_max_length=int(args.source_max_length),
        target_max_length=int(args.target_max_length),
        source_hidden_layer=int(args.source_hidden_layer),
        local_files_only=str(args.local_files_only).lower() == "true",
        prefix_len=int(args.prefix_len),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        bootstrap_samples=int(args.bootstrap_samples),
        continuation_mode=str(args.continuation_mode),
        matched_use_target=str(args.matched_use_target).lower() == "true",
        length_normalize=str(args.length_normalize).lower() == "true",
        same_byte_budget=int(args.same_byte_budget),
        min_accuracy_gap=float(args.min_accuracy_gap),
        min_margin_gap=float(args.min_margin_gap),
    )
    print(
        json.dumps(
            {
                "pass_gate": payload["pass_gate"],
                "fit_rows": payload["fit_rows"],
                "eval_rows": payload["eval_rows"],
                "matched_accuracy": payload["headline"]["matched_accuracy"],
                "best_control_by_accuracy": payload["headline"]["best_control_by_accuracy"],
                "best_control_accuracy": payload["headline"]["best_control_accuracy"],
                "matched_minus_best_control_margin": payload["headline"]["matched_minus_best_control_margin"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
