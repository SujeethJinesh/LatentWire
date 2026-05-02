#!/usr/bin/env python3
"""Cross-fit a source-conditioned soft-prefix logprob gate on SVAMP32.

This is a strict pre-generation diagnostic. It trains a tiny connector that
maps frozen source/target features into target input-embedding prefix tokens,
then optimizes target-model gold-vs-distractor continuation logprob directly.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from latent_bridge.evaluate import (
    _extract_prediction_numeric_answer,
    _extract_reference_numeric_answer,
    _format_prompt_for_tokenizer,
    _generation_example_id,
    load_generation,
)
from scripts import analyze_svamp32_target_query_source_bottleneck_probe as tq_probe
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome
from scripts import analyze_svamp32_source_latent_syndrome_probe as linear_probe
from scripts import analyze_svamp32_learned_syndrome_probe as learned_probe


@dataclass(frozen=True)
class SoftPrefixConfig:
    prefix_len: int = 4
    hidden_dim: int = 32
    epochs: int = 8
    lr: float = 3e-3
    weight_decay: float = 1e-3
    seed: int = 1
    outer_folds: str = "8"
    shuffle_offset: int = 1
    min_matched_only_clean: int = 2
    min_margin_delta: float = 0.0
    matched_use_target: bool = True
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
        use_source: bool = True,
        use_target: bool = True,
    ) -> None:
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.target_embed_dim = int(target_embed_dim)
        self.use_source = bool(use_source)
        self.use_target = bool(use_target)
        input_dim = 0
        if self.use_source:
            input_dim += int(source_dim)
        if self.use_target:
            input_dim += int(target_dim)
        if input_dim == 0:
            self.slots = torch.nn.Parameter(
                torch.randn(self.prefix_len, self.target_embed_dim) * 0.02
            )
            self.net = None
        else:
            self.slots = None
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, int(hidden_dim)),
                torch.nn.Tanh(),
                torch.nn.Linear(int(hidden_dim), self.prefix_len * self.target_embed_dim),
            )

    def forward(self, source_summary: torch.Tensor, target_summary: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            return self.slots
        parts: list[torch.Tensor] = []
        if self.use_source:
            parts.append(source_summary)
        if self.use_target:
            parts.append(target_summary)
        vector = torch.cat(parts, dim=-1)
        return self.net(vector).view(self.prefix_len, self.target_embed_dim)


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _method_records(path: pathlib.Path, method: str) -> list[dict[str, Any]]:
    rows = [row for row in _read_jsonl(path) if str(row.get("method")) == method]
    if rows:
        return rows
    alias = "c2c" if method == "c2c_generate" else method
    rows = [row for row in _read_jsonl(path) if str(row.get("method")) == alias]
    if not rows:
        available = sorted({str(row.get("method")) for row in _read_jsonl(path)})
        raise KeyError(f"method {method!r} not found in {path}; available={available}")
    return rows


def _by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row["example_id"])
        if example_id in out:
            raise ValueError(f"duplicate example_id={example_id}")
        out[example_id] = dict(row)
    return out


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    return (tokens * weights[:, :, None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)


def _standardize_matrix_for_fold(matrix: torch.Tensor, train_indices: Sequence[int]) -> torch.Tensor:
    train = matrix[list(train_indices)]
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (matrix - mean) / std


def _numeric_answer(answers: Sequence[str]) -> str:
    for answer in answers:
        numeric = _extract_reference_numeric_answer(str(answer))
        if numeric is not None:
            return str(numeric)
    raise ValueError(f"could not extract numeric answer from {answers!r}")


def _answer_continuation(answer: str, template: str) -> str:
    if "{answer}" not in template:
        raise ValueError("continuation template must contain {answer}")
    return template.format(answer=answer)


def _prediction_numeric(row: dict[str, Any]) -> str | None:
    for field in ("normalized_prediction", "prediction"):
        value = row.get(field)
        if value is None:
            continue
        numeric = _extract_prediction_numeric_answer(str(value))
        if numeric is not None:
            return str(numeric)
        numeric = _extract_reference_numeric_answer(str(value))
        if numeric is not None:
            return str(numeric)
    return None


def _continuation_ids(tokenizer: Any, text: str, device: str) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.shape[1] == 0:
        raise ValueError(f"zero-token continuation: {text!r}")
    return ids[0].to(device)


def _prompt_ids(tokenizer: Any, prompt: str, *, use_chat_template: bool, enable_thinking: bool | None, device: str) -> torch.Tensor:
    formatted = _format_prompt_for_tokenizer(
        tokenizer,
        prompt,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    )
    return tokenizer(formatted, return_tensors="pt").input_ids[0].to(device)


def _continuation_logprob(
    *,
    target_model: Any,
    embed_tokens: Any,
    prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    length_normalize: bool,
) -> torch.Tensor:
    prompt_embeds = embed_tokens(prompt_ids)
    continuation_embeds = embed_tokens(continuation_ids)
    if continuation_embeds.shape[0] > 1:
        inputs = torch.cat([prefix, prompt_embeds, continuation_embeds[:-1]], dim=0)
    else:
        inputs = torch.cat([prefix, prompt_embeds], dim=0)
    attention_mask = torch.ones((1, inputs.shape[0]), dtype=torch.long, device=inputs.device)
    out = target_model(inputs_embeds=inputs.unsqueeze(0), attention_mask=attention_mask, use_cache=False)
    logits = out.logits[0]
    start = int(prefix.shape[0] + prompt_embeds.shape[0] - 1)
    token_logits = logits[start : start + continuation_ids.shape[0]]
    logprobs = torch.log_softmax(token_logits.float(), dim=-1)
    total = logprobs.gather(1, continuation_ids[:, None]).sum()
    if length_normalize:
        return total / max(int(continuation_ids.shape[0]), 1)
    return total


def _example_answers(
    *,
    examples: Sequence[Any],
    target_records: Sequence[dict[str, Any]],
    teacher_records: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    target_by_id = _by_id(target_records)
    teacher_by_id = _by_id(teacher_records)
    rows: list[dict[str, str]] = []
    for example in examples:
        example_id = _generation_example_id(example)
        gold = _numeric_answer(example.answers)
        target = target_by_id[example_id]
        teacher = teacher_by_id[example_id]
        distractor = _prediction_numeric(target) or ""
        if not distractor or distractor == gold:
            distractor = _prediction_numeric(teacher) or ""
        if not distractor or distractor == gold:
            distractor = "0" if gold != "0" else "1"
        rows.append({"example_id": example_id, "gold": gold, "distractor": distractor})
    return rows


def _fit_connector(
    *,
    connector: SourceSoftPrefixConnector,
    target_model: Any,
    embed_tokens: Any,
    source_summary: torch.Tensor,
    target_summary: torch.Tensor,
    prompt_ids: Sequence[torch.Tensor],
    gold_ids: Sequence[torch.Tensor],
    distractor_ids: Sequence[torch.Tensor],
    train_indices: Sequence[int],
    config: SoftPrefixConfig,
    device: str,
    label_shuffle: bool = False,
) -> SourceSoftPrefixConnector:
    connector.to(device)
    optimizer = torch.optim.AdamW(
        connector.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay)
    )
    train_order = list(train_indices)
    if label_shuffle:
        shuffled_gold = list(train_order[::-1])
    else:
        shuffled_gold = list(train_order)
    for _ in range(int(config.epochs)):
        total_loss = torch.zeros((), device=device)
        for idx, gold_idx in zip(train_order, shuffled_gold):
            prefix = connector(source_summary[idx].to(device), target_summary[idx].to(device))
            gold_logprob = _continuation_logprob(
                target_model=target_model,
                embed_tokens=embed_tokens,
                prefix=prefix,
                prompt_ids=prompt_ids[idx],
                continuation_ids=gold_ids[gold_idx],
                length_normalize=config.length_normalize,
            )
            distractor_logprob = _continuation_logprob(
                target_model=target_model,
                embed_tokens=embed_tokens,
                prefix=prefix,
                prompt_ids=prompt_ids[idx],
                continuation_ids=distractor_ids[idx],
                length_normalize=config.length_normalize,
            )
            total_loss = total_loss + torch.nn.functional.softplus(
                -(gold_logprob - distractor_logprob)
            )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
    return connector.eval()


@torch.no_grad()
def _score_condition(
    *,
    connector: SourceSoftPrefixConnector,
    target_model: Any,
    embed_tokens: Any,
    source_summary: torch.Tensor,
    target_summary: torch.Tensor,
    prompt_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    distractor_ids: torch.Tensor,
    length_normalize: bool,
    device: str,
) -> dict[str, float]:
    prefix = connector(source_summary.to(device), target_summary.to(device))
    gold_logprob = _continuation_logprob(
        target_model=target_model,
        embed_tokens=embed_tokens,
        prefix=prefix,
        prompt_ids=prompt_ids,
        continuation_ids=gold_ids,
        length_normalize=length_normalize,
    )
    distractor_logprob = _continuation_logprob(
        target_model=target_model,
        embed_tokens=embed_tokens,
        prefix=prefix,
        prompt_ids=prompt_ids,
        continuation_ids=distractor_ids,
        length_normalize=length_normalize,
    )
    return {
        "gold_logprob": float(gold_logprob.item()),
        "distractor_logprob": float(distractor_logprob.item()),
        "margin": float((gold_logprob - distractor_logprob).item()),
    }


def _summarize(
    rows: Sequence[dict[str, Any]],
    *,
    clean_ids: set[str],
    target_self_ids: set[str],
    min_margin_delta: float,
) -> dict[str, Any]:
    clean_rows = [row for row in rows if row["example_id"] in clean_ids]
    target_self_rows = [row for row in rows if row["example_id"] in target_self_ids]
    matched_only = [
        row
        for row in clean_rows
        if row["matched_margin"] > 0.0
        and row["matched_minus_best_control_margin"] > min_margin_delta
    ]
    control_leak = [
        row
        for row in clean_rows
        if row["best_control_margin"] > 0.0
        and row["matched_margin"] <= row["best_control_margin"] + min_margin_delta
    ]
    return {
        "clean_ids_scored": len(clean_rows),
        "target_self_ids_scored": len(target_self_rows),
        "matched_positive_clean_count": sum(row["matched_margin"] > 0.0 for row in clean_rows),
        "matched_only_clean_count": len(matched_only),
        "matched_only_clean_ids": [row["example_id"] for row in matched_only],
        "control_leak_clean_count": len(control_leak),
        "control_leak_clean_ids": [row["example_id"] for row in control_leak],
        "target_self_matched_positive_count": sum(row["matched_margin"] > 0.0 for row in target_self_rows),
        "mean_matched_margin_clean": sum(row["matched_margin"] for row in clean_rows) / max(len(clean_rows), 1),
        "mean_best_control_margin_clean": sum(row["best_control_margin"] for row in clean_rows) / max(len(clean_rows), 1),
        "mean_matched_minus_best_control_clean": sum(row["matched_minus_best_control_margin"] for row in clean_rows) / max(len(clean_rows), 1),
    }


def _target_set_ids(target_set: dict[str, Any], *keys: str) -> list[str]:
    ids = target_set.get("ids", {})
    for key in keys:
        values = ids.get(key, [])
        if values:
            return [str(value) for value in values]
    return []


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# SVAMP32 Source Soft-Prefix Logprob Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- prefix len: `{payload['config']['prefix_len']}`",
        f"- hidden dim: `{payload['config']['hidden_dim']}`",
        f"- epochs: `{payload['config']['epochs']}`",
        f"- clean IDs scored: `{summary['clean_ids_scored']}`",
        f"- matched-only clean IDs: `{summary['matched_only_clean_count']}`",
        f"- control-leak clean IDs: `{summary['control_leak_clean_count']}`",
        f"- mean matched-minus-control clean: `{summary['mean_matched_minus_best_control_clean']:.6f}`",
        "",
        "## Clean Rows",
        "",
        "| Example ID | Gold | Distractor | Matched Margin | Best Control | Best Control Margin | Delta | Status |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        if "clean_source_communication_candidate" not in row["labels"]:
            continue
        lines.append(
            "| {example_id} | {gold} | {distractor} | {matched:.6f} | {best_control} | {control:.6f} | {delta:.6f} | {status} |".format(
                example_id=row["example_id"],
                gold=row["gold_answer"],
                distractor=row["distractor_answer"],
                matched=row["matched_margin"],
                best_control=row["best_control"],
                control=row["best_control_margin"],
                delta=row["matched_minus_best_control_margin"],
                status=row["status"],
            )
        )
    lines.extend(["", "## Controls", ""])
    lines.append(
        "`matched`, `zero_source`, `shuffled_source`, `same_norm_noise`, "
        "`target_only_prefix`, `slots_only_prefix`, `label_shuffled`, and "
        "`projected_soft_prompt` are scored for every clean row."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _bool_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target-jsonl", required=True)
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--target-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--target-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--feature-layers", default="mid,last")
    parser.add_argument("--prefix-len", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-folds", default="8")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--continuation-template", default=" {answer}")
    parser.add_argument("--min-matched-only-clean", type=int, default=2)
    parser.add_argument("--min-margin-delta", type=float, default=0.0)
    parser.add_argument("--matched-use-target", choices=["true", "false"], default="true")
    parser.add_argument("--length-normalize", choices=["true", "false"], default="true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    device = str(args.device)
    train_device = str(args.train_device or args.device)
    examples = load_generation(str(_resolve(args.eval_file)))
    target_records = _method_records(_resolve(args.target_jsonl), "target_alone")
    teacher_records = _method_records(_resolve(args.teacher_jsonl), "c2c_generate")
    reference_ids = [str(row["example_id"]) for row in target_records]
    example_ids = [_generation_example_id(example) for example in examples]
    if reference_ids != example_ids:
        raise ValueError("target JSONL IDs must match eval-file ordered IDs")
    target_set = _read_json(_resolve(args.target_set_json))
    clean_ids = _target_set_ids(
        target_set,
        "clean_residual_targets",
        "clean_teacher_only",
        "clean_c2c_headroom_targets",
    )
    target_self_ids = _target_set_ids(target_set, "target_self_repair", "target_correct")
    if not target_self_ids:
        target_self_ids = [
            str(row["example_id"]) for row in target_records if bool(row.get("correct"))
        ]
    answers = _example_answers(
        examples=examples,
        target_records=target_records,
        teacher_records=teacher_records,
    )
    config = SoftPrefixConfig(
        prefix_len=int(args.prefix_len),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        outer_folds=str(args.outer_folds),
        shuffle_offset=int(args.shuffle_offset),
        min_matched_only_clean=int(args.min_matched_only_clean),
        min_margin_delta=float(args.min_margin_delta),
        matched_use_target=_bool_arg(args.matched_use_target) is not False,
        length_normalize=_bool_arg(args.length_normalize) is not False,
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading source model: {args.source_model}", flush=True)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if source_tokenizer.pad_token_id is None:
        source_tokenizer.pad_token = source_tokenizer.eos_token
    source_model = (
        AutoModelForCausalLM.from_pretrained(
            args.source_model,
            torch_dtype=linear_probe._torch_dtype(args.dtype),
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    source_tokens, source_mask, _ = tq_probe._extract_token_features(
        model=source_model,
        tokenizer=source_tokenizer,
        examples=examples,
        device=device,
        reasoning_mode=str(args.source_reasoning_mode),
        use_chat_template=bool(args.source_use_chat_template),
        enable_thinking=_bool_arg(args.source_enable_thinking),
        feature_layers=str(args.feature_layers),
        source_style=True,
    )
    del source_model

    print(f"Loading target model: {args.target_model}", flush=True)
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if target_tokenizer.pad_token_id is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    target_model = (
        AutoModelForCausalLM.from_pretrained(
            args.target_model,
            torch_dtype=linear_probe._torch_dtype(args.dtype),
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    for param in target_model.parameters():
        param.requires_grad_(False)
    embed_tokens = target_model.get_input_embeddings()
    target_tokens, target_mask, _ = tq_probe._extract_token_features(
        model=target_model,
        tokenizer=target_tokenizer,
        examples=examples,
        device=device,
        reasoning_mode=str(args.source_reasoning_mode),
        use_chat_template=bool(args.target_use_chat_template),
        enable_thinking=_bool_arg(args.target_enable_thinking),
        feature_layers=str(args.feature_layers),
        source_style=False,
    )

    source_summary = _masked_mean(source_tokens.float(), source_mask).to(train_device)
    target_summary = _masked_mean(target_tokens.float(), target_mask).to(train_device)
    prompt_ids = [
        _prompt_ids(
            target_tokenizer,
            example.prompt,
            use_chat_template=bool(args.target_use_chat_template),
            enable_thinking=_bool_arg(args.target_enable_thinking),
            device=train_device,
        )
        for example in examples
    ]
    gold_ids = [
        _continuation_ids(
            target_tokenizer,
            _answer_continuation(row["gold"], str(args.continuation_template)),
            train_device,
        )
        for row in answers
    ]
    distractor_ids = [
        _continuation_ids(
            target_tokenizer,
            _answer_continuation(row["distractor"], str(args.continuation_template)),
            train_device,
        )
        for row in answers
    ]
    target_model.to(train_device)
    embed_tokens = target_model.get_input_embeddings()

    torch.manual_seed(int(args.seed))
    folds = learned_probe._make_outer_folds(len(examples), config.outer_folds)
    rows: list[dict[str, Any]] = []
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_indices = [idx for idx in range(len(examples)) if idx not in heldout_set]
        fold_source_summary = _standardize_matrix_for_fold(source_summary, train_indices).to(train_device)
        fold_target_summary = _standardize_matrix_for_fold(target_summary, train_indices).to(train_device)
        source_dim = int(fold_source_summary.shape[-1])
        target_dim = int(fold_target_summary.shape[-1])
        embed_dim = int(embed_tokens.embedding_dim)
        matched = SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            prefix_len=config.prefix_len,
            use_source=True,
            use_target=config.matched_use_target,
        )
        target_only = SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            prefix_len=config.prefix_len,
            use_source=False,
            use_target=True,
        )
        slots_only = SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            prefix_len=config.prefix_len,
            use_source=False,
            use_target=False,
        )
        label_shuffled = SourceSoftPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            prefix_len=config.prefix_len,
            use_source=True,
            use_target=True,
        )
        matched = _fit_connector(
            connector=matched,
            target_model=target_model,
            embed_tokens=embed_tokens,
            source_summary=fold_source_summary,
            target_summary=fold_target_summary,
            prompt_ids=prompt_ids,
            gold_ids=gold_ids,
            distractor_ids=distractor_ids,
            train_indices=train_indices,
            config=config,
            device=train_device,
            label_shuffle=False,
        )
        target_only = _fit_connector(
            connector=target_only,
            target_model=target_model,
            embed_tokens=embed_tokens,
            source_summary=fold_source_summary,
            target_summary=fold_target_summary,
            prompt_ids=prompt_ids,
            gold_ids=gold_ids,
            distractor_ids=distractor_ids,
            train_indices=train_indices,
            config=config,
            device=train_device,
            label_shuffle=False,
        )
        slots_only = _fit_connector(
            connector=slots_only,
            target_model=target_model,
            embed_tokens=embed_tokens,
            source_summary=fold_source_summary,
            target_summary=fold_target_summary,
            prompt_ids=prompt_ids,
            gold_ids=gold_ids,
            distractor_ids=distractor_ids,
            train_indices=train_indices,
            config=config,
            device=train_device,
            label_shuffle=False,
        )
        label_shuffled = _fit_connector(
            connector=label_shuffled,
            target_model=target_model,
            embed_tokens=embed_tokens,
            source_summary=fold_source_summary,
            target_summary=fold_target_summary,
            prompt_ids=prompt_ids,
            gold_ids=gold_ids,
            distractor_ids=distractor_ids,
            train_indices=train_indices,
            config=config,
            device=train_device,
            label_shuffle=True,
        )
        train_mean_source = fold_source_summary[train_indices].mean(dim=0)
        for idx in heldout:
            shuffled_idx = (idx + int(config.shuffle_offset)) % len(examples)
            generator = torch.Generator().manual_seed(int(config.seed) * 1009 + idx)
            same_norm = torch.randn(fold_source_summary[idx].shape, generator=generator).to(train_device)
            same_norm = same_norm / same_norm.norm().clamp_min(1e-6) * fold_source_summary[idx].norm().clamp_min(1e-6)
            condition_scores = {
                "matched": _score_condition(
                    connector=matched,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_summary=fold_source_summary[idx],
                    target_summary=fold_target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    gold_ids=gold_ids[idx],
                    distractor_ids=distractor_ids[idx],
                    length_normalize=config.length_normalize,
                    device=train_device,
                ),
                "zero_source": _score_condition(
                    connector=matched,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_summary=torch.zeros_like(fold_source_summary[idx]),
                    target_summary=fold_target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    gold_ids=gold_ids[idx],
                    distractor_ids=distractor_ids[idx],
                    length_normalize=config.length_normalize,
                    device=train_device,
                ),
                "shuffled_source": _score_condition(
                    connector=matched,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_summary=fold_source_summary[shuffled_idx],
                    target_summary=fold_target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    gold_ids=gold_ids[idx],
                    distractor_ids=distractor_ids[idx],
                    length_normalize=config.length_normalize,
                    device=train_device,
                ),
                "same_norm_noise": _score_condition(
                    connector=matched,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_summary=same_norm,
                    target_summary=fold_target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    gold_ids=gold_ids[idx],
                    distractor_ids=distractor_ids[idx],
                    length_normalize=config.length_normalize,
                    device=train_device,
                ),
                "projected_soft_prompt": _score_condition(
                    connector=matched,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_summary=train_mean_source,
                    target_summary=fold_target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    gold_ids=gold_ids[idx],
                    distractor_ids=distractor_ids[idx],
                    length_normalize=config.length_normalize,
                    device=train_device,
                ),
                "target_only_prefix": _score_condition(
                    connector=target_only,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_summary=torch.zeros_like(fold_source_summary[idx]),
                    target_summary=fold_target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    gold_ids=gold_ids[idx],
                    distractor_ids=distractor_ids[idx],
                    length_normalize=config.length_normalize,
                    device=train_device,
                ),
                "slots_only_prefix": _score_condition(
                    connector=slots_only,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_summary=torch.zeros_like(fold_source_summary[idx]),
                    target_summary=torch.zeros_like(fold_target_summary[idx]),
                    prompt_ids=prompt_ids[idx],
                    gold_ids=gold_ids[idx],
                    distractor_ids=distractor_ids[idx],
                    length_normalize=config.length_normalize,
                    device=train_device,
                ),
                "label_shuffled": _score_condition(
                    connector=label_shuffled,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_summary=fold_source_summary[idx],
                    target_summary=fold_target_summary[idx],
                    prompt_ids=prompt_ids[idx],
                    gold_ids=gold_ids[idx],
                    distractor_ids=distractor_ids[idx],
                    length_normalize=config.length_normalize,
                    device=train_device,
                ),
            }
            matched_margin = condition_scores["matched"]["margin"]
            controls = {
                name: score["margin"]
                for name, score in condition_scores.items()
                if name != "matched"
            }
            best_control, best_margin = max(controls.items(), key=lambda item: item[1])
            labels = []
            if reference_ids[idx] in clean_ids:
                labels.append("clean_source_communication_candidate")
            if reference_ids[idx] in target_self_ids:
                labels.append("target_self_repair")
            delta = float(matched_margin - best_margin)
            rows.append(
                {
                    "index": idx,
                    "fold": fold_idx,
                    "example_id": reference_ids[idx],
                    "labels": labels,
                    "gold_answer": answers[idx]["gold"],
                    "distractor_answer": answers[idx]["distractor"],
                    "matched_margin": float(matched_margin),
                    "best_control": best_control,
                    "best_control_margin": float(best_margin),
                    "matched_minus_best_control_margin": delta,
                    "status": (
                        "matched_only_positive"
                        if matched_margin > 0.0 and delta > config.min_margin_delta
                        else "control_or_negative"
                    ),
                    "scores": condition_scores,
                }
            )
            print(
                f"{reference_ids[idx]} fold={fold_idx} matched={matched_margin:.4f} "
                f"best={best_control}:{best_margin:.4f} delta={delta:.4f}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["index"]))
    summary = _summarize(
        rows,
        clean_ids=set(clean_ids),
        target_self_ids=set(target_self_ids),
        min_margin_delta=config.min_margin_delta,
    )
    status = (
        "source_soft_prefix_logprob_candidate"
        if summary["matched_only_clean_count"] >= config.min_matched_only_clean
        and summary["control_leak_clean_count"] == 0
        else "source_soft_prefix_logprob_fails_gate"
    )
    payload = {
        "date": str(args.date),
        "status": status,
        "reference_n": len(reference_ids),
        "command": " ".join(sys.argv),
        "artifacts": {
            "eval_file": _display(_resolve(args.eval_file)),
            "target_jsonl": _display(_resolve(args.target_jsonl)),
            "teacher_jsonl": _display(_resolve(args.teacher_jsonl)),
            "target_set_json": _display(_resolve(args.target_set_json)),
        },
        "config": {
            "source_model": args.source_model,
            "target_model": args.target_model,
            "feature_layers": str(args.feature_layers),
            "prefix_len": config.prefix_len,
            "hidden_dim": config.hidden_dim,
            "epochs": config.epochs,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
            "outer_folds": config.outer_folds,
            "shuffle_offset": config.shuffle_offset,
            "min_matched_only_clean": config.min_matched_only_clean,
            "min_margin_delta": config.min_margin_delta,
            "matched_use_target": config.matched_use_target,
            "length_normalize": config.length_normalize,
        },
        "ids": {
            "clean_source_communication_candidates": clean_ids,
            "target_self_or_target_correct": target_self_ids,
        },
        "summary": summary,
        "rows": rows,
    }
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": status, "output_json": _display(output_json)}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
