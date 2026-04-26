#!/usr/bin/env python3
"""Cross-fit a token-local source cross-attention logprob gate on SVAMP32.

This is the next strict pre-generation diagnostic after global summary
soft-prefix connectors failed. It lets target-conditioned learned prefix
queries attend over standardized source token states, then scores target-model
gold-vs-distractor continuation logprob under source-destroying and
target-only controls.
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

from latent_bridge.evaluate import _generation_example_id, load_generation
from scripts import analyze_svamp32_learned_syndrome_probe as learned_probe
from scripts import analyze_svamp32_source_latent_syndrome_probe as linear_probe
from scripts import analyze_svamp32_source_soft_prefix_logprob_probe as soft_probe
from scripts import analyze_svamp32_target_query_source_bottleneck_probe as tq_probe


@dataclass(frozen=True)
class CrossAttentionConfig:
    prefix_len: int = 4
    hidden_dim: int = 32
    epochs: int = 4
    lr: float = 3e-3
    weight_decay: float = 1e-3
    seed: int = 1
    outer_folds: str = "8"
    shuffle_offset: int = 1
    min_matched_only_clean: int = 2
    min_margin_delta: float = 0.0
    length_normalize: bool = True


class SourceCrossAttentionPrefixConnector(torch.nn.Module):
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
        if not self.use_source and not self.use_target:
            self.slots = torch.nn.Parameter(
                torch.randn(self.prefix_len, self.target_embed_dim) * 0.02
            )
            self.source_proj = None
            self.target_proj = None
            self.target_to_prefix = None
            self.target_to_query_delta = None
            self.queries = None
            self.out = None
            return
        self.slots = None
        if self.use_source:
            self.source_proj = torch.nn.Linear(int(source_dim), int(hidden_dim))
            self.queries = torch.nn.Parameter(torch.randn(self.prefix_len, int(hidden_dim)) * 0.02)
            self.out = torch.nn.Linear(int(hidden_dim), self.target_embed_dim)
            if self.use_target:
                self.target_proj = torch.nn.Linear(int(target_dim), int(hidden_dim))
                self.target_to_query_delta = torch.nn.Linear(
                    int(hidden_dim), self.prefix_len * int(hidden_dim)
                )
            else:
                self.target_proj = None
                self.target_to_query_delta = None
            self.target_to_prefix = None
        else:
            self.source_proj = None
            self.queries = None
            self.out = None
            self.target_proj = torch.nn.Linear(int(target_dim), int(hidden_dim))
            self.target_to_prefix = torch.nn.Sequential(
                torch.nn.Tanh(),
                torch.nn.Linear(int(hidden_dim), self.prefix_len * self.target_embed_dim),
            )
            self.target_to_query_delta = None

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float()
        return (tokens * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1.0)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.slots is not None:
            return self.slots
        if not self.use_source:
            target_summary = self._masked_mean(target_tokens, target_mask)
            hidden = self.target_proj(target_summary)
            return self.target_to_prefix(hidden).view(self.prefix_len, self.target_embed_dim)

        source_hidden = torch.tanh(self.source_proj(source_tokens))
        queries = self.queries
        if self.use_target:
            target_summary = self._masked_mean(target_tokens, target_mask)
            target_hidden = torch.tanh(self.target_proj(target_summary))
            query_delta = self.target_to_query_delta(target_hidden).view_as(queries)
            queries = queries + query_delta
        scores = torch.einsum("qh,th->qt", queries, source_hidden) / math.sqrt(
            source_hidden.shape[-1]
        )
        scores = scores.masked_fill(~source_mask[None, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("qt,th->qh", weights, source_hidden)
        return self.out(context)


def _standardized_tokens_for_fold(
    tokens: torch.Tensor, mask: torch.Tensor, train_indices: Sequence[int]
) -> torch.Tensor:
    return learned_probe._standardize_tokens_for_fold(tokens, mask, train_indices)


def _fit_connector(
    *,
    connector: SourceCrossAttentionPrefixConnector,
    target_model: Any,
    embed_tokens: Any,
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    prompt_ids: Sequence[torch.Tensor],
    gold_ids: Sequence[torch.Tensor],
    distractor_ids: Sequence[torch.Tensor],
    train_indices: Sequence[int],
    config: CrossAttentionConfig,
    device: str,
    label_shuffle: bool = False,
) -> SourceCrossAttentionPrefixConnector:
    connector.to(device)
    optimizer = torch.optim.AdamW(
        connector.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay)
    )
    train_order = list(train_indices)
    shuffled_gold = list(train_order[::-1]) if label_shuffle else list(train_order)
    for _ in range(int(config.epochs)):
        total_loss = torch.zeros((), device=device)
        for idx, gold_idx in zip(train_order, shuffled_gold):
            prefix = connector(
                source_tokens[idx].to(device),
                source_mask[idx].to(device),
                target_tokens[idx].to(device),
                target_mask[idx].to(device),
            )
            gold_logprob = soft_probe._continuation_logprob(
                target_model=target_model,
                embed_tokens=embed_tokens,
                prefix=prefix,
                prompt_ids=prompt_ids[idx],
                continuation_ids=gold_ids[gold_idx],
                length_normalize=config.length_normalize,
            )
            distractor_logprob = soft_probe._continuation_logprob(
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
    connector: SourceCrossAttentionPrefixConnector,
    target_model: Any,
    embed_tokens: Any,
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
    gold_ids: torch.Tensor,
    distractor_ids: torch.Tensor,
    length_normalize: bool,
    device: str,
) -> dict[str, float]:
    prefix = connector(
        source_tokens.to(device),
        source_mask.to(device),
        target_tokens.to(device),
        target_mask.to(device),
    )
    gold_logprob = soft_probe._continuation_logprob(
        target_model=target_model,
        embed_tokens=embed_tokens,
        prefix=prefix,
        prompt_ids=prompt_ids,
        continuation_ids=gold_ids,
        length_normalize=length_normalize,
    )
    distractor_logprob = soft_probe._continuation_logprob(
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


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# SVAMP32 Source Cross-Attention Logprob Probe",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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
    parser.add_argument("--feature-layers", default="last")
    parser.add_argument("--prefix-len", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-folds", default="2")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--continuation-template", default=" {answer}")
    parser.add_argument("--min-matched-only-clean", type=int, default=2)
    parser.add_argument("--min-margin-delta", type=float, default=0.0)
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
    examples = load_generation(str(soft_probe._resolve(args.eval_file)))
    target_records = soft_probe._method_records(soft_probe._resolve(args.target_jsonl), "target_alone")
    teacher_records = soft_probe._method_records(soft_probe._resolve(args.teacher_jsonl), "c2c_generate")
    reference_ids = [str(row["example_id"]) for row in target_records]
    example_ids = [_generation_example_id(example) for example in examples]
    if reference_ids != example_ids:
        raise ValueError("target JSONL IDs must match eval-file ordered IDs")

    target_set = soft_probe._read_json(soft_probe._resolve(args.target_set_json))
    clean_ids = soft_probe._target_set_ids(
        target_set,
        "clean_residual_targets",
        "clean_teacher_only",
        "clean_c2c_headroom_targets",
    )
    target_self_ids = soft_probe._target_set_ids(target_set, "target_self_repair", "target_correct")
    if not target_self_ids:
        target_self_ids = [
            str(row["example_id"]) for row in target_records if bool(row.get("correct"))
        ]
    answers = soft_probe._example_answers(
        examples=examples,
        target_records=target_records,
        teacher_records=teacher_records,
    )
    config = CrossAttentionConfig(
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
        length_normalize=soft_probe._bool_arg(args.length_normalize) is not False,
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
        enable_thinking=soft_probe._bool_arg(args.source_enable_thinking),
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
        enable_thinking=soft_probe._bool_arg(args.target_enable_thinking),
        feature_layers=str(args.feature_layers),
        source_style=False,
    )

    prompt_ids = [
        soft_probe._prompt_ids(
            target_tokenizer,
            example.prompt,
            use_chat_template=bool(args.target_use_chat_template),
            enable_thinking=soft_probe._bool_arg(args.target_enable_thinking),
            device=train_device,
        )
        for example in examples
    ]
    gold_ids = [
        soft_probe._continuation_ids(
            target_tokenizer,
            soft_probe._answer_continuation(row["gold"], str(args.continuation_template)),
            train_device,
        )
        for row in answers
    ]
    distractor_ids = [
        soft_probe._continuation_ids(
            target_tokenizer,
            soft_probe._answer_continuation(row["distractor"], str(args.continuation_template)),
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
        fold_source_tokens = _standardized_tokens_for_fold(source_tokens, source_mask, train_indices)
        fold_target_tokens = _standardized_tokens_for_fold(target_tokens, target_mask, train_indices)
        source_dim = int(fold_source_tokens.shape[-1])
        target_dim = int(fold_target_tokens.shape[-1])
        embed_dim = int(embed_tokens.embedding_dim)
        matched = SourceCrossAttentionPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            prefix_len=config.prefix_len,
            use_source=True,
            use_target=True,
        )
        target_only = SourceCrossAttentionPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            prefix_len=config.prefix_len,
            use_source=False,
            use_target=True,
        )
        slots_only = SourceCrossAttentionPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            prefix_len=config.prefix_len,
            use_source=False,
            use_target=False,
        )
        label_shuffled = SourceCrossAttentionPrefixConnector(
            source_dim=source_dim,
            target_dim=target_dim,
            target_embed_dim=embed_dim,
            hidden_dim=config.hidden_dim,
            prefix_len=config.prefix_len,
            use_source=True,
            use_target=True,
        )
        for connector, label_shuffle in (
            (matched, False),
            (target_only, False),
            (slots_only, False),
            (label_shuffled, True),
        ):
            _fit_connector(
                connector=connector,
                target_model=target_model,
                embed_tokens=embed_tokens,
                source_tokens=fold_source_tokens,
                source_mask=source_mask,
                target_tokens=fold_target_tokens,
                target_mask=target_mask,
                prompt_ids=prompt_ids,
                gold_ids=gold_ids,
                distractor_ids=distractor_ids,
                train_indices=train_indices,
                config=config,
                device=train_device,
                label_shuffle=label_shuffle,
            )

        train_valid = fold_source_tokens[train_indices][source_mask[train_indices]]
        train_mean_source = train_valid.mean(dim=0)
        for idx in heldout:
            shuffled_idx = (idx + int(config.shuffle_offset)) % len(examples)
            generator = torch.Generator().manual_seed(int(config.seed) * 1009 + idx)
            same_norm = torch.randn(fold_source_tokens[idx].shape, generator=generator)
            same_norm = same_norm.to(train_device)
            same_norm = same_norm / same_norm.norm().clamp_min(1e-6) * fold_source_tokens[idx].norm().clamp_min(1e-6)
            projected = train_mean_source[None, :].repeat(fold_source_tokens[idx].shape[0], 1)
            condition_scores = {
                "matched": _score_condition(
                    connector=matched,
                    target_model=target_model,
                    embed_tokens=embed_tokens,
                    source_tokens=fold_source_tokens[idx],
                    source_mask=source_mask[idx],
                    target_tokens=fold_target_tokens[idx],
                    target_mask=target_mask[idx],
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
                    source_tokens=torch.zeros_like(fold_source_tokens[idx]),
                    source_mask=source_mask[idx],
                    target_tokens=fold_target_tokens[idx],
                    target_mask=target_mask[idx],
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
                    source_tokens=fold_source_tokens[shuffled_idx],
                    source_mask=source_mask[shuffled_idx],
                    target_tokens=fold_target_tokens[idx],
                    target_mask=target_mask[idx],
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
                    source_tokens=same_norm,
                    source_mask=source_mask[idx],
                    target_tokens=fold_target_tokens[idx],
                    target_mask=target_mask[idx],
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
                    source_tokens=projected,
                    source_mask=source_mask[idx],
                    target_tokens=fold_target_tokens[idx],
                    target_mask=target_mask[idx],
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
                    source_tokens=torch.zeros_like(fold_source_tokens[idx]),
                    source_mask=source_mask[idx],
                    target_tokens=fold_target_tokens[idx],
                    target_mask=target_mask[idx],
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
                    source_tokens=torch.zeros_like(fold_source_tokens[idx]),
                    source_mask=source_mask[idx],
                    target_tokens=torch.zeros_like(fold_target_tokens[idx]),
                    target_mask=target_mask[idx],
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
                    source_tokens=fold_source_tokens[idx],
                    source_mask=source_mask[idx],
                    target_tokens=fold_target_tokens[idx],
                    target_mask=target_mask[idx],
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
    summary = soft_probe._summarize(
        rows,
        clean_ids=set(clean_ids),
        target_self_ids=set(target_self_ids),
        min_margin_delta=config.min_margin_delta,
    )
    status = (
        "source_cross_attention_logprob_candidate"
        if summary["matched_only_clean_count"] >= config.min_matched_only_clean
        and summary["control_leak_clean_count"] == 0
        else "source_cross_attention_logprob_fails_gate"
    )
    payload = {
        "date": str(args.date),
        "status": status,
        "reference_n": len(reference_ids),
        "command": " ".join(sys.argv),
        "artifacts": {
            "eval_file": soft_probe._display(soft_probe._resolve(args.eval_file)),
            "target_jsonl": soft_probe._display(soft_probe._resolve(args.target_jsonl)),
            "teacher_jsonl": soft_probe._display(soft_probe._resolve(args.teacher_jsonl)),
            "target_set_json": soft_probe._display(soft_probe._resolve(args.target_set_json)),
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
            "length_normalize": config.length_normalize,
        },
        "ids": {
            "clean_source_communication_candidates": clean_ids,
            "target_self_or_target_correct": target_self_ids,
        },
        "summary": summary,
        "rows": rows,
    }
    output_json = soft_probe._resolve(args.output_json)
    output_md = soft_probe._resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": status, "output_json": soft_probe._display(output_json)}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
