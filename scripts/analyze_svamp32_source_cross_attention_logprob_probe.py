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
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from latent_bridge.evaluate import _generation_example_id, load_generation
from scripts import harness_common as harness
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
    training_objective: str = "contrastive"
    source_control_contrastive_weight: float = 0.0
    source_control_contrastive_margin: float = 0.0
    source_control_contrastive_controls: tuple[str, ...] = ()


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
    train_valid = source_tokens[train_indices][source_mask[train_indices]]
    train_mean_source = train_valid.mean(dim=0) if train_valid.numel() else None

    def _gold_logprob_for_prefix(prefix: torch.Tensor, idx: int, gold_idx: int) -> torch.Tensor:
        return soft_probe._continuation_logprob(
            target_model=target_model,
            embed_tokens=embed_tokens,
            prefix=prefix,
            prompt_ids=prompt_ids[idx],
            continuation_ids=gold_ids[gold_idx],
            length_normalize=config.length_normalize,
        )

    def _margin_for_prefix(prefix: torch.Tensor, idx: int, gold_idx: int) -> torch.Tensor:
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
        return gold_logprob - distractor_logprob

    for _ in range(int(config.epochs)):
        total_loss = torch.zeros((), device=device)
        for idx, gold_idx in zip(train_order, shuffled_gold):
            prefix = connector(
                source_tokens[idx].to(device),
                source_mask[idx].to(device),
                target_tokens[idx].to(device),
                target_mask[idx].to(device),
            )
            matched_margin = _margin_for_prefix(prefix, idx, gold_idx)
            if config.training_objective == "target_ce":
                matched_gold_logprob = _gold_logprob_for_prefix(prefix, idx, gold_idx)
                total_loss = total_loss - matched_gold_logprob
            elif config.training_objective == "contrastive":
                matched_gold_logprob = None
                total_loss = total_loss + torch.nn.functional.softplus(-matched_margin)
            else:
                raise ValueError(f"unsupported training objective: {config.training_objective!r}")
            if (
                connector.use_source
                and not label_shuffle
                and config.source_control_contrastive_weight > 0.0
                and config.source_control_contrastive_controls
            ):
                controls: list[tuple[torch.Tensor, torch.Tensor]] = []
                for control in config.source_control_contrastive_controls:
                    if control == "zero_source":
                        controls.append((torch.zeros_like(source_tokens[idx]), source_mask[idx]))
                    elif control == "shuffled_source":
                        shuffled_idx = (idx + int(config.shuffle_offset)) % len(source_tokens)
                        controls.append((source_tokens[shuffled_idx], source_mask[shuffled_idx]))
                    elif control == "same_norm_noise":
                        noise = torch.randn_like(source_tokens[idx])
                        noise = noise / noise.norm().clamp_min(1e-6) * source_tokens[idx].norm().clamp_min(1e-6)
                        controls.append((noise, source_mask[idx]))
                    elif control == "projected_soft_prompt":
                        if train_mean_source is None:
                            continue
                        projected = train_mean_source[None, :].repeat(source_tokens[idx].shape[0], 1)
                        controls.append((projected, source_mask[idx]))
                    else:
                        raise ValueError(f"unsupported source control contrastive control: {control!r}")
                for control_tokens, control_mask in controls:
                    control_prefix = connector(
                        control_tokens.to(device),
                        control_mask.to(device),
                        target_tokens[idx].to(device),
                        target_mask[idx].to(device),
                    )
                    if config.training_objective == "target_ce":
                        if matched_gold_logprob is None:
                            raise AssertionError("matched gold logprob missing for target_ce")
                        control_gold_logprob = _gold_logprob_for_prefix(control_prefix, idx, gold_idx)
                        contrastive_term = control_gold_logprob - matched_gold_logprob
                    else:
                        control_margin = _margin_for_prefix(control_prefix, idx, gold_idx)
                        contrastive_term = control_margin - matched_margin
                    total_loss = total_loss + float(
                        config.source_control_contrastive_weight
                    ) * torch.nn.functional.softplus(
                        contrastive_term + float(config.source_control_contrastive_margin)
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


@torch.no_grad()
def _greedy_generate_with_prefix(
    *,
    target_model: Any,
    embed_tokens: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt_embeds = embed_tokens(prompt_ids)
    inputs = torch.cat([prefix, prompt_embeds], dim=0)
    attention_mask = torch.ones((1, inputs.shape[0]), dtype=torch.long, device=inputs.device)
    start = time.perf_counter()
    out = target_model(inputs_embeds=inputs.unsqueeze(0), attention_mask=attention_mask, use_cache=True)
    first_elapsed = time.perf_counter() - start
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
    generated: list[int] = []
    eos_ids = {int(token_id) for token_id in [tokenizer.eos_token_id, tokenizer.pad_token_id] if token_id is not None}
    for _ in range(int(max_new_tokens)):
        token_id = int(next_token.item())
        if token_id in eos_ids:
            break
        generated.append(token_id)
        step_input = next_token.view(1, 1)
        attention_mask = torch.ones(
            (1, int(inputs.shape[0]) + len(generated)),
            dtype=torch.long,
            device=inputs.device,
        )
        out = target_model(
            input_ids=step_input,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
    elapsed = time.perf_counter() - start
    return {
        "prediction": tokenizer.decode(generated, skip_special_tokens=True),
        "generated_tokens": len(generated),
        "ttft_sec": float(first_elapsed),
        "latency_sec": float(elapsed),
    }


def _generation_record(
    *,
    method: str,
    example_id: str,
    index: int,
    generation: dict[str, Any],
    answers: Sequence[str],
) -> dict[str, Any]:
    prediction = str(generation["prediction"])
    normalized = harness._extract_prediction_numeric_answer(prediction)
    return {
        "method": method,
        "example_id": example_id,
        "index": int(index),
        "prediction": prediction,
        "normalized_prediction": normalized,
        "correct": harness._generation_match(prediction, list(answers)),
        "generated_tokens": int(generation["generated_tokens"]),
        "ttft_sec": float(generation["ttft_sec"]),
        "latency_sec": float(generation["latency_sec"]),
    }


def _summarize_generation(
    records_by_condition: dict[str, list[dict[str, Any]]],
    *,
    clean_ids: set[str],
    target_self_ids: set[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    correct_by_condition = {
        condition: {str(row["example_id"]) for row in records if bool(row.get("correct"))}
        for condition, records in records_by_condition.items()
    }
    for condition, records in records_by_condition.items():
        correct_ids = correct_by_condition[condition]
        summary[condition] = {
            "n": len(records),
            "correct_count": len(correct_ids),
            "accuracy": float(len(correct_ids) / max(len(records), 1)),
            "clean_correct_count": len(correct_ids & clean_ids),
            "target_self_correct_count": len(correct_ids & target_self_ids),
            "numeric_extraction_coverage": int(
                sum(int(row.get("normalized_prediction") is not None) for row in records)
            ),
            "empty_predictions": int(sum(int(not str(row.get("prediction", "")).strip()) for row in records)),
            "correct_ids": sorted(correct_ids),
        }
    matched_correct = correct_by_condition.get("matched", set())
    control_correct = set().union(
        *[ids for condition, ids in correct_by_condition.items() if condition != "matched"]
    ) if len(correct_by_condition) > 1 else set()
    matched_only_clean = (matched_correct - control_correct) & clean_ids
    control_leak_clean = (control_correct - matched_correct) & clean_ids
    summary["gate"] = {
        "matched_only_clean_count": len(matched_only_clean),
        "control_leak_clean_count": len(control_leak_clean),
        "matched_only_clean_ids": sorted(matched_only_clean),
        "control_leak_clean_ids": sorted(control_leak_clean),
    }
    return summary


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
        f"- training objective: `{payload['config'].get('training_objective', 'contrastive')}`",
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
    generation_summary = payload.get("generation_summary") or {}
    if generation_summary:
        gate = generation_summary.get("gate", {})
        lines.extend(
            [
                "",
                "## Generation Gate",
                "",
                f"- matched-only clean IDs: `{gate.get('matched_only_clean_count', 0)}`",
                f"- control-leak clean IDs: `{gate.get('control_leak_clean_count', 0)}`",
                "",
                "| Condition | Correct | Clean Correct | Target-Self Correct | Numeric Coverage | Empty |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, condition_summary in generation_summary.items():
            if condition == "gate":
                continue
            lines.append(
                "| {condition} | {correct} | {clean} | {target_self} | {numeric} | {empty} |".format(
                    condition=condition,
                    correct=condition_summary["correct_count"],
                    clean=condition_summary["clean_correct_count"],
                    target_self=condition_summary["target_self_correct_count"],
                    numeric=condition_summary["numeric_extraction_coverage"],
                    empty=condition_summary["empty_predictions"],
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
    parser.add_argument(
        "--training-objective",
        choices=["contrastive", "target_ce"],
        default="contrastive",
        help="Connector training loss. target_ce is true target-side continuation next-token NLL.",
    )
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--generation-example-id",
        action="append",
        default=[],
        help="Restrict generation decoding to specific example IDs while still scoring logprob on all heldout rows.",
    )
    parser.add_argument(
        "--generation-condition",
        action="append",
        choices=[
            "matched",
            "zero_source",
            "shuffled_source",
            "same_norm_noise",
            "projected_soft_prompt",
            "target_only_prefix",
            "slots_only_prefix",
            "label_shuffled",
        ],
        default=[],
        help="Heldout condition to decode. Defaults to matched plus source/target-only controls when --run-generation is set.",
    )
    parser.add_argument("--generation-output-jsonl", default=None)
    parser.add_argument("--source-control-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--source-control-contrastive-margin", type=float, default=0.0)
    parser.add_argument(
        "--source-control-contrastive-control",
        action="append",
        choices=["zero_source", "shuffled_source", "same_norm_noise", "projected_soft_prompt"],
        default=[],
        help=(
            "Matched-source training penalty control. The matched connector is "
            "penalized when this source-destroying condition matches or beats "
            "the real source margin."
        ),
    )
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
        training_objective=str(args.training_objective),
        source_control_contrastive_weight=float(args.source_control_contrastive_weight),
        source_control_contrastive_margin=float(args.source_control_contrastive_margin),
        source_control_contrastive_controls=tuple(args.source_control_contrastive_control),
    )
    generation_conditions = tuple(
        args.generation_condition
        or ["matched", "zero_source", "shuffled_source", "target_only_prefix", "slots_only_prefix"]
    )
    generation_example_ids = {str(example_id) for example_id in args.generation_example_id}
    if args.run_generation and not args.generation_output_jsonl:
        raise ValueError("--generation-output-jsonl is required when --run-generation is set")

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
    generation_records_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in generation_conditions
    }
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
            generation_by_condition: dict[str, dict[str, Any]] = {}
            if args.run_generation and (not generation_example_ids or reference_ids[idx] in generation_example_ids):
                def _prefix_for_generation(condition: str) -> torch.Tensor:
                    if condition == "matched":
                        return matched(
                            fold_source_tokens[idx].to(train_device),
                            source_mask[idx].to(train_device),
                            fold_target_tokens[idx].to(train_device),
                            target_mask[idx].to(train_device),
                        )
                    if condition == "zero_source":
                        return matched(
                            torch.zeros_like(fold_source_tokens[idx]).to(train_device),
                            source_mask[idx].to(train_device),
                            fold_target_tokens[idx].to(train_device),
                            target_mask[idx].to(train_device),
                        )
                    if condition == "shuffled_source":
                        return matched(
                            fold_source_tokens[shuffled_idx].to(train_device),
                            source_mask[shuffled_idx].to(train_device),
                            fold_target_tokens[idx].to(train_device),
                            target_mask[idx].to(train_device),
                        )
                    if condition == "same_norm_noise":
                        return matched(
                            same_norm.to(train_device),
                            source_mask[idx].to(train_device),
                            fold_target_tokens[idx].to(train_device),
                            target_mask[idx].to(train_device),
                        )
                    if condition == "projected_soft_prompt":
                        return matched(
                            projected.to(train_device),
                            source_mask[idx].to(train_device),
                            fold_target_tokens[idx].to(train_device),
                            target_mask[idx].to(train_device),
                        )
                    if condition == "target_only_prefix":
                        return target_only(
                            torch.zeros_like(fold_source_tokens[idx]).to(train_device),
                            source_mask[idx].to(train_device),
                            fold_target_tokens[idx].to(train_device),
                            target_mask[idx].to(train_device),
                        )
                    if condition == "slots_only_prefix":
                        return slots_only(
                            torch.zeros_like(fold_source_tokens[idx]).to(train_device),
                            source_mask[idx].to(train_device),
                            torch.zeros_like(fold_target_tokens[idx]).to(train_device),
                            target_mask[idx].to(train_device),
                        )
                    if condition == "label_shuffled":
                        return label_shuffled(
                            fold_source_tokens[idx].to(train_device),
                            source_mask[idx].to(train_device),
                            fold_target_tokens[idx].to(train_device),
                            target_mask[idx].to(train_device),
                        )
                    raise ValueError(f"unsupported generation condition: {condition!r}")

                for condition in generation_conditions:
                    prefix = _prefix_for_generation(condition)
                    generated = _greedy_generate_with_prefix(
                        target_model=target_model,
                        embed_tokens=embed_tokens,
                        tokenizer=target_tokenizer,
                        prefix=prefix,
                        prompt_ids=prompt_ids[idx],
                        max_new_tokens=int(args.generation_max_new_tokens),
                    )
                    record = _generation_record(
                        method=f"source_cross_attention_{config.training_objective}_{condition}",
                        example_id=reference_ids[idx],
                        index=idx,
                        generation=generated,
                        answers=examples[idx].answers,
                    )
                    generation_records_by_condition[condition].append(record)
                    generation_by_condition[condition] = {
                        "prediction": record["prediction"],
                        "normalized_prediction": record["normalized_prediction"],
                        "correct": record["correct"],
                        "generated_tokens": record["generated_tokens"],
                        "ttft_sec": record["ttft_sec"],
                        "latency_sec": record["latency_sec"],
                    }
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
                    "generation": generation_by_condition,
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
    generation_summary = (
        _summarize_generation(
            generation_records_by_condition,
            clean_ids=set(clean_ids),
            target_self_ids=set(target_self_ids),
        )
        if args.run_generation
        else {}
    )
    status = (
        "source_cross_attention_generation_candidate"
        if args.run_generation
        and generation_summary.get("gate", {}).get("matched_only_clean_count", 0)
        >= config.min_matched_only_clean
        and generation_summary.get("gate", {}).get("control_leak_clean_count", 0) == 0
        else (
            "source_cross_attention_logprob_candidate"
            if summary["matched_only_clean_count"] >= config.min_matched_only_clean
            and summary["control_leak_clean_count"] == 0
            else "source_cross_attention_logprob_fails_gate"
        )
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
            "generation_output_jsonl": (
                soft_probe._display(soft_probe._resolve(args.generation_output_jsonl))
                if args.generation_output_jsonl
                else None
            ),
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
            "training_objective": config.training_objective,
            "run_generation": bool(args.run_generation),
            "generation_max_new_tokens": int(args.generation_max_new_tokens),
            "generation_conditions": list(generation_conditions),
            "generation_example_ids": sorted(generation_example_ids),
            "source_control_contrastive_weight": config.source_control_contrastive_weight,
            "source_control_contrastive_margin": config.source_control_contrastive_margin,
            "source_control_contrastive_controls": list(config.source_control_contrastive_controls),
        },
        "ids": {
            "clean_source_communication_candidates": clean_ids,
            "target_self_or_target_correct": target_self_ids,
        },
        "summary": summary,
        "generation_summary": generation_summary,
        "rows": rows,
    }
    output_json = soft_probe._resolve(args.output_json)
    output_md = soft_probe._resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.run_generation:
        generation_output_jsonl = soft_probe._resolve(args.generation_output_jsonl)
        generation_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with generation_output_jsonl.open("w", encoding="utf-8") as handle:
            for condition in generation_conditions:
                for record in generation_records_by_condition[condition]:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
    _write_markdown(output_md, payload)
    print(json.dumps({"status": status, "output_json": soft_probe._display(output_json)}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    main()
