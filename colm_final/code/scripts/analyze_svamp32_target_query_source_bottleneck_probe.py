#!/usr/bin/env python3
"""Cross-fit a target-query-conditioned source bottleneck on SVAMP32 rows.

This is a strict diagnostic, not a generation method. It asks whether target
prompt states can query source token states to predict C2C residue classes
better than source-destroying and source-independent controls.
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
    _format_prompt_for_tokenizer,
    _generation_example_id,
    _source_reasoning_prompt,
    load_generation,
)
from scripts import analyze_svamp32_learned_syndrome_probe as learned_probe
from scripts import analyze_svamp32_source_latent_syndrome_probe as linear_probe
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


@dataclass(frozen=True)
class TargetQueryConfig:
    moduli: tuple[int, ...]
    query_count: int = 4
    hidden_dim: int = 16
    epochs: int = 80
    lr: float = 3e-3
    weight_decay: float = 1e-3
    seed: int = 1
    outer_folds: str = "8"
    shuffle_offset: int = 1
    min_correct: int = 10
    min_clean_source_necessary: int = 2


class TargetQuerySourceBottleneck(torch.nn.Module):
    def __init__(
        self,
        *,
        source_dim: int,
        target_dim: int,
        hidden_dim: int,
        query_count: int,
        moduli: Sequence[int],
    ) -> None:
        super().__init__()
        self.moduli = tuple(int(value) for value in moduli)
        self.source_proj = torch.nn.Linear(source_dim, hidden_dim)
        self.target_proj = torch.nn.Linear(target_dim, hidden_dim)
        self.learned_queries = torch.nn.Parameter(
            torch.randn(query_count, hidden_dim) * 0.02
        )
        self.heads = torch.nn.ModuleDict(
            {str(modulus): torch.nn.Linear(hidden_dim, modulus) for modulus in self.moduli}
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_mask: torch.Tensor,
        target_tokens: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        source_hidden = torch.tanh(self.source_proj(source_tokens))
        target_hidden = torch.tanh(self.target_proj(target_tokens))
        target_weights = target_mask.float()
        target_summary = (
            target_hidden * target_weights[:, :, None]
        ).sum(dim=1) / target_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        queries = self.learned_queries[None, :, :] + target_summary[:, None, :]
        scores = torch.einsum("bqh,bth->bqt", queries, source_hidden) / math.sqrt(
            source_hidden.shape[-1]
        )
        scores = scores.masked_fill(~source_mask[:, None, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bqt,bth->bqh", weights, source_hidden).mean(dim=1)
        return {modulus: self.heads[str(modulus)](pooled) for modulus in self.moduli}


@torch.no_grad()
def _extract_token_features(
    *,
    model,
    tokenizer,
    examples,
    device: str,
    reasoning_mode: str,
    use_chat_template: bool,
    enable_thinking: bool | None,
    feature_layers: str,
    source_style: bool,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    rows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for example in examples:
        prompt = (
            _source_reasoning_prompt(example.prompt, reasoning_mode)
            if source_style
            else example.prompt
        )
        formatted = _format_prompt_for_tokenizer(
            tokenizer,
            prompt,
            use_chat_template=use_chat_template,
            enable_thinking=enable_thinking,
        )
        input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to(device)
        out = model(
            input_ids=input_ids,
            attention_mask=linear_probe._ones(input_ids.shape[1], device),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = tuple(out.hidden_states)
        layer_indices = linear_probe._feature_layer_indices(
            len(hidden_states), feature_layers
        )
        token_features = torch.cat(
            [hidden_states[layer_idx][0].detach().float().cpu() for layer_idx in layer_indices],
            dim=-1,
        )
        rows.append(token_features)
        metadata.append(
            {
                "example_id": _generation_example_id(example),
                "formatted_prompt_tokens": int(input_ids.shape[1]),
                "feature_dim": int(token_features.shape[-1]),
                "feature_layers": layer_indices,
                "source_style": bool(source_style),
            }
        )
    max_len = max(int(row.shape[0]) for row in rows)
    feature_dim = int(rows[0].shape[-1])
    tokens = torch.zeros((len(rows), max_len, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool)
    for idx, row in enumerate(rows):
        tokens[idx, : row.shape[0], :] = row
        mask[idx, : row.shape[0]] = True
    return tokens, mask, metadata


def _label_tensors_for_indices(
    labels_by_modulus: dict[int, torch.Tensor],
    indices: Sequence[int],
    *,
    label_shuffle: bool,
    device: str,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    index_tensor = torch.tensor(list(indices), dtype=torch.long)
    for modulus, labels in labels_by_modulus.items():
        values = labels[index_tensor]
        if label_shuffle:
            values = values[torch.arange(values.numel() - 1, -1, -1)]
        result[modulus] = values.to(device)
    return result


def _fit_target_query_model(
    *,
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    labels_by_modulus: dict[int, torch.Tensor],
    train_indices: Sequence[int],
    config: TargetQueryConfig,
    device: str,
    label_shuffle: bool,
) -> TargetQuerySourceBottleneck:
    torch.manual_seed(int(config.seed) + (991 if label_shuffle else 0))
    model = TargetQuerySourceBottleneck(
        source_dim=int(source_tokens.shape[-1]),
        target_dim=int(target_tokens.shape[-1]),
        hidden_dim=int(config.hidden_dim),
        query_count=int(config.query_count),
        moduli=config.moduli,
    ).to(device)
    train_source = source_tokens[list(train_indices)].to(device)
    train_source_mask = source_mask[list(train_indices)].to(device)
    train_target = target_tokens[list(train_indices)].to(device)
    train_target_mask = target_mask[list(train_indices)].to(device)
    train_labels = _label_tensors_for_indices(
        labels_by_modulus,
        train_indices,
        label_shuffle=label_shuffle,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay)
    )
    for _ in range(int(config.epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_source, train_source_mask, train_target, train_target_mask)
        loss = sum(
            torch.nn.functional.cross_entropy(logits[modulus], train_labels[modulus])
            for modulus in config.moduli
        )
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def _predict_signature(
    model: TargetQuerySourceBottleneck,
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    device: str,
) -> tuple[int, ...]:
    logits = model(
        source_tokens[None, ...].to(device),
        source_mask[None, ...].to(device),
        target_tokens[None, ...].to(device),
        target_mask[None, ...].to(device),
    )
    return tuple(int(torch.argmax(logits[modulus], dim=-1).item()) for modulus in model.moduli)


def _crossfit_target_query_predictions(
    *,
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    labels_by_modulus: dict[int, torch.Tensor],
    config: TargetQueryConfig,
    device: str,
) -> tuple[dict[str, list[tuple[int, ...]]], list[dict[str, Any]]]:
    n = int(source_tokens.shape[0])
    folds = learned_probe._make_outer_folds(n, config.outer_folds)
    predictions: dict[str, list[tuple[int, ...] | None]] = {
        "matched": [None] * n,
        "zero_source": [None] * n,
        "shuffled_source": [None] * n,
        "label_shuffled": [None] * n,
        "same_norm_noise": [None] * n,
        "target_only_prefix": [None] * n,
        "projected_soft_prompt": [None] * n,
    }
    fold_metadata: list[dict[str, Any]] = []
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_indices = [idx for idx in range(n) if idx not in heldout_set]
        fold_source = learned_probe._standardize_tokens_for_fold(
            source_tokens, source_mask, train_indices
        )
        fold_target = learned_probe._standardize_tokens_for_fold(
            target_tokens, target_mask, train_indices
        )
        matched_model = _fit_target_query_model(
            source_tokens=fold_source,
            source_mask=source_mask,
            target_tokens=fold_target,
            target_mask=target_mask,
            labels_by_modulus=labels_by_modulus,
            train_indices=train_indices,
            config=config,
            device=device,
            label_shuffle=False,
        )
        label_model = _fit_target_query_model(
            source_tokens=fold_source,
            source_mask=source_mask,
            target_tokens=fold_target,
            target_mask=target_mask,
            labels_by_modulus=labels_by_modulus,
            train_indices=train_indices,
            config=config,
            device=device,
            label_shuffle=True,
        )
        train_valid = fold_source[list(train_indices)][source_mask[list(train_indices)]]
        source_mean = train_valid.mean(dim=0)
        for idx in heldout:
            shuffled_idx = (idx + int(config.shuffle_offset)) % n
            if shuffled_idx == idx:
                shuffled_idx = (idx + 1) % n
            zero_source = torch.zeros_like(fold_source[idx])
            generator = torch.Generator().manual_seed(int(config.seed) * 1009 + idx)
            mean_source = torch.zeros_like(fold_source[idx])
            mean_source[source_mask[idx]] = source_mean
            predictions["matched"][idx] = _predict_signature(
                matched_model,
                fold_source[idx],
                source_mask[idx],
                fold_target[idx],
                target_mask[idx],
                device=device,
            )
            predictions["zero_source"][idx] = _predict_signature(
                matched_model,
                zero_source,
                source_mask[idx],
                fold_target[idx],
                target_mask[idx],
                device=device,
            )
            predictions["target_only_prefix"][idx] = predictions["zero_source"][idx]
            predictions["shuffled_source"][idx] = _predict_signature(
                matched_model,
                fold_source[shuffled_idx],
                source_mask[shuffled_idx],
                fold_target[idx],
                target_mask[idx],
                device=device,
            )
            predictions["label_shuffled"][idx] = _predict_signature(
                label_model,
                fold_source[idx],
                source_mask[idx],
                fold_target[idx],
                target_mask[idx],
                device=device,
            )
            predictions["same_norm_noise"][idx] = _predict_signature(
                matched_model,
                learned_probe._same_norm_noise(
                    fold_source[idx], source_mask[idx], generator=generator
                ),
                source_mask[idx],
                fold_target[idx],
                target_mask[idx],
                device=device,
            )
            predictions["projected_soft_prompt"][idx] = _predict_signature(
                matched_model,
                mean_source,
                source_mask[idx],
                fold_target[idx],
                target_mask[idx],
                device=device,
            )
        fold_metadata.append(
            {
                "fold_index": fold_idx,
                "heldout_indices": list(heldout),
                "train_indices": train_indices,
            }
        )
    finalized: dict[str, list[tuple[int, ...]]] = {}
    for condition, values in predictions.items():
        if any(value is None for value in values):
            raise RuntimeError(f"missing cross-fit predictions for {condition}")
        finalized[condition] = [tuple(value) for value in values if value is not None]
    return finalized, fold_metadata


def analyze_with_token_features(
    *,
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    target_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    source_metadata: Sequence[dict[str, Any]],
    target_metadata: Sequence[dict[str, Any]],
    target_spec: syndrome.RowSpec,
    teacher_spec: syndrome.RowSpec,
    candidate_specs: Sequence[syndrome.RowSpec],
    target_set_path: pathlib.Path,
    fallback_label: str,
    config: TargetQueryConfig,
    min_numeric_coverage: int,
    run_date: str,
    device: str,
) -> dict[str, Any]:
    target_records = syndrome._records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("target rows contain duplicate example_id values")
    source_ids = [str(row["example_id"]) for row in source_metadata]
    target_feature_ids = [str(row["example_id"]) for row in target_metadata]
    if source_ids != reference_ids or target_feature_ids != reference_ids:
        raise ValueError("feature metadata IDs do not match target ordered IDs")
    teacher_records = syndrome._subset_reference_order(
        syndrome._records_for_method(teacher_spec), reference_ids
    )
    target_by_id = syndrome._by_id(target_records)
    teacher_by_id = syndrome._by_id(teacher_records)
    target_ids = syndrome._load_target_ids(target_set_path)
    if target_ids["teacher_only"]:
        target_correct = {
            str(row["example_id"]) for row in target_records if bool(row.get("correct"))
        }
        teacher_correct = {
            str(row["example_id"]) for row in teacher_records if bool(row.get("correct"))
        }
        if target_ids["teacher_only"] != teacher_correct - target_correct:
            raise ValueError("target_set.ids.teacher_only does not match target/teacher rows")

    candidate_by_label: dict[str, dict[str, dict[str, Any]]] = {
        target_spec.label: target_by_id
    }
    for spec in candidate_specs:
        if spec.label in candidate_by_label:
            raise ValueError(f"Duplicate candidate label {spec.label!r}")
        candidate_by_label[spec.label] = syndrome._by_id(
            syndrome._subset_reference_order(
                syndrome._records_for_method(spec), reference_ids
            )
        )
    if fallback_label not in candidate_by_label:
        raise ValueError(f"fallback label {fallback_label!r} missing from candidate labels")

    teacher_numeric_coverage = sum(
        int(syndrome._prediction_numeric(row) is not None) for row in teacher_records
    )
    candidate_numeric_coverage = {
        label: sum(
            int(syndrome._prediction_numeric(records[example_id]) is not None)
            for example_id in reference_ids
        )
        for label, records in candidate_by_label.items()
    }
    provenance_issues: list[str] = []
    if teacher_numeric_coverage < min_numeric_coverage:
        provenance_issues.append(
            f"teacher_numeric_coverage={teacher_numeric_coverage} < {min_numeric_coverage}"
        )
    for label, coverage in candidate_numeric_coverage.items():
        if coverage < min_numeric_coverage:
            provenance_issues.append(
                f"candidate.{label}.numeric_coverage={coverage} < {min_numeric_coverage}"
            )

    labels_by_modulus = linear_probe._teacher_residue_labels(
        teacher_records, config.moduli
    )
    residue_predictions, fold_metadata = _crossfit_target_query_predictions(
        source_tokens=source_tokens.float().cpu(),
        source_mask=source_mask.cpu(),
        target_tokens=target_tokens.float().cpu(),
        target_mask=target_mask.cpu(),
        labels_by_modulus=labels_by_modulus,
        config=config,
        device=device,
    )
    learned_config = learned_probe.LearnedProbeConfig(
        moduli=config.moduli,
        query_count=config.query_count,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        seed=config.seed,
        outer_folds=config.outer_folds,
        shuffle_offset=config.shuffle_offset,
        min_correct=config.min_correct,
        min_clean_source_necessary=config.min_clean_source_necessary,
    )
    run = learned_probe._evaluate_learned_probe(
        reference_ids=reference_ids,
        target_by_id=target_by_id,
        teacher_by_id=teacher_by_id,
        candidate_by_label=candidate_by_label,
        target_label=target_spec.label,
        fallback_label=fallback_label,
        target_ids=target_ids,
        residue_predictions=residue_predictions,
        config=learned_config,
    )
    status = (
        run["status"].replace("learned_syndrome_probe", "target_query_source_bottleneck")
        if not provenance_issues
        else "target_query_source_bottleneck_fails_gate"
    )
    return {
        "date": run_date,
        "status": status,
        "interpretation": (
            "This probe trains a cross-fitted target-query-conditioned "
            "source bottleneck. Target prompt states form queries over "
            "source token states to predict C2C residue classes. It is a "
            "strict pre-generation diagnostic, not a generation method."
        ),
        "artifacts": {
            "target": {
                "label": target_spec.label,
                "path": linear_probe._display_path(target_spec.path),
                "method": target_spec.method,
            },
            "teacher": {
                "label": teacher_spec.label,
                "path": linear_probe._display_path(teacher_spec.path),
                "method": teacher_spec.method,
            },
            "target_set_json": linear_probe._display_path(target_set_path),
            "candidates": [
                {
                    "label": spec.label,
                    "path": linear_probe._display_path(spec.path),
                    "method": spec.method,
                }
                for spec in candidate_specs
            ],
        },
        "config": {
            "fallback_label": fallback_label,
            "moduli": list(config.moduli),
            "query_count": int(config.query_count),
            "hidden_dim": int(config.hidden_dim),
            "epochs": int(config.epochs),
            "lr": float(config.lr),
            "weight_decay": float(config.weight_decay),
            "seed": int(config.seed),
            "outer_folds": config.outer_folds,
            "shuffle_offset": int(config.shuffle_offset),
            "min_correct": int(config.min_correct),
            "min_clean_source_necessary": int(config.min_clean_source_necessary),
            "min_numeric_coverage": int(min_numeric_coverage),
            "target_query_conditioned": True,
            "source_independent_controls": [
                "target_only_prefix",
                "projected_soft_prompt",
            ],
        },
        "reference_n": len(reference_ids),
        "target_ids": {key: sorted(value) for key, value in target_ids.items()},
        "feature_metadata": list(source_metadata),
        "target_feature_metadata": list(target_metadata),
        "fold_metadata": fold_metadata,
        "provenance": {
            "exact_ordered_id_parity": True,
            "teacher_numeric_coverage": teacher_numeric_coverage,
            "candidate_numeric_coverage": candidate_numeric_coverage,
            "issues": provenance_issues,
        },
        "run": run,
    }


def _write_markdown(path: pathlib.Path, payload: dict[str, Any]) -> None:
    run = payload["run"]
    lines = [
        "# SVAMP32 Target-Query Source Bottleneck Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- moduli: `{','.join(str(value) for value in payload['config']['moduli'])}`",
        f"- query count: `{payload['config']['query_count']}`",
        f"- hidden dim: `{payload['config']['hidden_dim']}`",
        f"- epochs: `{payload['config']['epochs']}`",
        f"- outer folds: `{payload['config']['outer_folds']}`",
        f"- teacher numeric coverage: `{payload['provenance']['teacher_numeric_coverage']}/{payload['reference_n']}`",
        f"- provenance issues: `{len(payload['provenance']['issues'])}`",
        "",
        "## Summary",
        "",
        "| Condition | Correct | Clean Correct | Target-Self Correct |",
        "|---|---:|---:|---:|",
    ]
    for condition, summary in run["condition_summaries"].items():
        lines.append(
            "| {condition} | {correct} | {clean} | {target_self} |".format(
                condition=condition,
                correct=summary["correct_count"],
                clean=summary["clean_correct_count"],
                target_self=summary["target_self_correct_count"],
            )
        )
    lines.extend(
        [
            "",
            f"- clean source-necessary IDs: `{len(run['source_necessary_clean_ids'])}`",
            f"- source-necessary IDs: {', '.join(f'`{value}`' for value in run['source_necessary_clean_ids']) or 'none'}",
            f"- control clean union IDs: {', '.join(f'`{value}`' for value in run['control_clean_union_ids']) or 'none'}",
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
        ]
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
    parser.add_argument("--target", required=True, type=syndrome._parse_spec)
    parser.add_argument("--teacher", required=True, type=syndrome._parse_spec)
    parser.add_argument("--candidate", action="append", type=syndrome._parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target")
    parser.add_argument("--moduli", default="2,3,5,7")
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-folds", default="8")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--min-correct", type=int, default=10)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--min-numeric-coverage", type=int, default=31)
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--target-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--target-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--feature-layers", default="mid,last")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--train-device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    examples = load_generation(str(linear_probe._resolve(args.eval_file)))
    moduli = tuple(int(value) for value in str(args.moduli).split(",") if value.strip())
    config = TargetQueryConfig(
        moduli=moduli,
        query_count=int(args.query_count),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        outer_folds=str(args.outer_folds),
        shuffle_offset=int(args.shuffle_offset),
        min_correct=int(args.min_correct),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
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
        .to(args.device)
        .eval()
    )
    source_tokens, source_mask, source_metadata = _extract_token_features(
        model=source_model,
        tokenizer=source_tokenizer,
        examples=examples,
        device=args.device,
        reasoning_mode=args.source_reasoning_mode,
        use_chat_template=bool(args.source_use_chat_template),
        enable_thinking=_bool_arg(args.source_enable_thinking),
        feature_layers=str(args.feature_layers),
        source_style=True,
    )
    del source_model
    if str(args.source_model) == str(args.target_model):
        target_tokenizer = source_tokenizer
    else:
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
        .to(args.device)
        .eval()
    )
    target_tokens, target_mask, target_metadata = _extract_token_features(
        model=target_model,
        tokenizer=target_tokenizer,
        examples=examples,
        device=args.device,
        reasoning_mode=args.source_reasoning_mode,
        use_chat_template=bool(args.target_use_chat_template),
        enable_thinking=_bool_arg(args.target_enable_thinking),
        feature_layers=str(args.feature_layers),
        source_style=False,
    )
    payload = analyze_with_token_features(
        source_tokens=source_tokens,
        source_mask=source_mask,
        target_tokens=target_tokens,
        target_mask=target_mask,
        source_metadata=source_metadata,
        target_metadata=target_metadata,
        target_spec=args.target,
        teacher_spec=args.teacher,
        candidate_specs=args.candidate,
        target_set_path=linear_probe._resolve(args.target_set_json),
        fallback_label=str(args.fallback_label),
        config=config,
        min_numeric_coverage=int(args.min_numeric_coverage),
        run_date=str(args.date),
        device=str(args.train_device or args.device),
    )
    output_json = linear_probe._resolve(args.output_json)
    output_md = linear_probe._resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_md, payload)
    print(
        json.dumps(
            {"status": payload["status"], "output_json": linear_probe._display_path(output_json)},
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
