#!/usr/bin/env python3
"""Cross-fit a tiny source-token syndrome predictor on frozen SVAMP32 rows.

This is a strict diagnostic. It keeps the target candidate-pool decoder from
the syndrome sidecar bound, but replaces pooled hidden-state ridge readout with
a small learned query bottleneck over source token states.
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
from scripts import analyze_svamp32_source_latent_syndrome_probe as linear_probe
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


@dataclass(frozen=True)
class LearnedProbeConfig:
    moduli: tuple[int, ...]
    query_count: int = 4
    hidden_dim: int = 16
    epochs: int = 80
    lr: float = 3e-3
    weight_decay: float = 1e-3
    seed: int = 1
    outer_folds: str = "8"
    shuffle_offset: int = 1
    min_correct: int = 14
    min_clean_source_necessary: int = 2


class SyndromeQ(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        query_count: int,
        moduli: Sequence[int],
    ) -> None:
        super().__init__()
        self.moduli = tuple(int(value) for value in moduli)
        self.token_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.queries = torch.nn.Parameter(torch.randn(query_count, hidden_dim) * 0.02)
        self.heads = torch.nn.ModuleDict(
            {str(modulus): torch.nn.Linear(hidden_dim, modulus) for modulus in self.moduli}
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[int, torch.Tensor]:
        hidden = torch.tanh(self.token_proj(tokens))
        scores = torch.einsum("qh,bth->bqt", self.queries, hidden) / math.sqrt(hidden.shape[-1])
        scores = scores.masked_fill(~mask[:, None, :], -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bqt,bth->bqh", weights, hidden).mean(dim=1)
        return {modulus: self.heads[str(modulus)](pooled) for modulus in self.moduli}


@torch.no_grad()
def _extract_source_token_features(
    *,
    model,
    tokenizer,
    examples,
    device: str,
    source_reasoning_mode: str,
    use_chat_template: bool,
    enable_thinking: bool | None,
    feature_layers: str,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    rows: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    for example in examples:
        prompt = _source_reasoning_prompt(example.prompt, source_reasoning_mode)
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
        layer_indices = linear_probe._feature_layer_indices(len(hidden_states), feature_layers)
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


def _make_outer_folds(n: int, spec: str) -> list[list[int]]:
    if spec == "loo":
        return [[idx] for idx in range(n)]
    fold_count = int(spec)
    if fold_count < 2 or fold_count > n:
        raise ValueError(f"outer_folds must be 'loo' or an integer in 2..{n}")
    folds = [[] for _ in range(fold_count)]
    for idx in range(n):
        folds[idx % fold_count].append(idx)
    return folds


def _standardize_tokens_for_fold(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    train_indices: Sequence[int],
) -> torch.Tensor:
    train_tokens = tokens[list(train_indices)]
    train_mask = mask[list(train_indices)]
    valid = train_tokens[train_mask]
    mean = valid.mean(dim=0, keepdim=True)
    std = valid.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (tokens - mean.view(1, 1, -1)) / std.view(1, 1, -1)


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


def _fit_syndrome_q(
    *,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    labels_by_modulus: dict[int, torch.Tensor],
    train_indices: Sequence[int],
    config: LearnedProbeConfig,
    device: str,
    label_shuffle: bool,
) -> SyndromeQ:
    torch.manual_seed(int(config.seed) + (991 if label_shuffle else 0))
    model = SyndromeQ(
        input_dim=int(tokens.shape[-1]),
        hidden_dim=int(config.hidden_dim),
        query_count=int(config.query_count),
        moduli=config.moduli,
    ).to(device)
    train_tokens = tokens[list(train_indices)].to(device)
    train_mask = mask[list(train_indices)].to(device)
    train_labels = _label_tensors_for_indices(
        labels_by_modulus,
        train_indices,
        label_shuffle=label_shuffle,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )
    for _ in range(int(config.epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tokens, train_mask)
        loss = sum(
            torch.nn.functional.cross_entropy(logits[modulus], train_labels[modulus])
            for modulus in config.moduli
        )
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def _predict_signature(
    model: SyndromeQ,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: str,
) -> tuple[int, ...]:
    logits = model(tokens[None, ...].to(device), mask[None, ...].to(device))
    return tuple(int(torch.argmax(logits[modulus], dim=-1).item()) for modulus in model.moduli)


def _same_norm_noise(
    source_tokens: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(source_tokens.shape, generator=generator, dtype=source_tokens.dtype)
    valid_norm = source_tokens[source_mask].norm().clamp_min(1e-6)
    noise_norm = noise[source_mask].norm().clamp_min(1e-6)
    return noise * (valid_norm / noise_norm)


def _crossfit_residue_predictions(
    *,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    labels_by_modulus: dict[int, torch.Tensor],
    config: LearnedProbeConfig,
    device: str,
) -> tuple[dict[str, list[tuple[int, ...]]], list[dict[str, Any]]]:
    n = int(tokens.shape[0])
    folds = _make_outer_folds(n, config.outer_folds)
    predictions: dict[str, list[tuple[int, ...] | None]] = {
        "matched": [None] * n,
        "zero_source": [None] * n,
        "shuffled_source": [None] * n,
        "label_shuffled": [None] * n,
        "same_norm_noise": [None] * n,
    }
    fold_metadata: list[dict[str, Any]] = []
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_indices = [idx for idx in range(n) if idx not in heldout_set]
        fold_tokens = _standardize_tokens_for_fold(tokens, mask, train_indices)
        matched_model = _fit_syndrome_q(
            tokens=fold_tokens,
            mask=mask,
            labels_by_modulus=labels_by_modulus,
            train_indices=train_indices,
            config=config,
            device=device,
            label_shuffle=False,
        )
        label_model = _fit_syndrome_q(
            tokens=fold_tokens,
            mask=mask,
            labels_by_modulus=labels_by_modulus,
            train_indices=train_indices,
            config=config,
            device=device,
            label_shuffle=True,
        )
        for idx in heldout:
            shuffled_idx = (idx + int(config.shuffle_offset)) % n
            if shuffled_idx == idx:
                shuffled_idx = (idx + 1) % n
            zero_tokens = torch.zeros_like(fold_tokens[idx])
            generator = torch.Generator().manual_seed(int(config.seed) * 1009 + idx)
            predictions["matched"][idx] = _predict_signature(
                matched_model,
                fold_tokens[idx],
                mask[idx],
                device=device,
            )
            predictions["zero_source"][idx] = _predict_signature(
                matched_model,
                zero_tokens,
                mask[idx],
                device=device,
            )
            predictions["shuffled_source"][idx] = _predict_signature(
                matched_model,
                fold_tokens[shuffled_idx],
                mask[shuffled_idx],
                device=device,
            )
            predictions["label_shuffled"][idx] = _predict_signature(
                label_model,
                fold_tokens[idx],
                mask[idx],
                device=device,
            )
            predictions["same_norm_noise"][idx] = _predict_signature(
                matched_model,
                _same_norm_noise(fold_tokens[idx], mask[idx], generator=generator),
                mask[idx],
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


def _summarize_condition(
    rows: Sequence[dict[str, Any]],
    *,
    condition: str,
    clean_ids: set[str],
    target_self_ids: set[str],
    teacher_only_ids: set[str],
) -> dict[str, Any]:
    correct_ids = {
        str(row["example_id"])
        for row in rows
        if bool(row["conditions"][condition]["correct"])
    }
    return {
        "condition": condition,
        "correct_count": len(correct_ids),
        "correct_ids": sorted(correct_ids),
        "clean_correct_count": len(correct_ids & clean_ids),
        "clean_correct_ids": sorted(correct_ids & clean_ids),
        "target_self_correct_count": len(correct_ids & target_self_ids),
        "target_self_correct_ids": sorted(correct_ids & target_self_ids),
        "teacher_only_correct_count": len(correct_ids & teacher_only_ids),
        "teacher_only_correct_ids": sorted(correct_ids & teacher_only_ids),
    }


def _evaluate_learned_probe(
    *,
    reference_ids: Sequence[str],
    target_by_id: dict[str, dict[str, Any]],
    teacher_by_id: dict[str, dict[str, Any]],
    candidate_by_label: dict[str, dict[str, dict[str, Any]]],
    target_label: str,
    fallback_label: str,
    target_ids: dict[str, set[str]],
    residue_predictions: dict[str, list[tuple[int, ...]]],
    config: LearnedProbeConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidate_labels = list(candidate_by_label)
    prediction_conditions = tuple(residue_predictions.keys())
    for index, example_id in enumerate(reference_ids):
        rows_by_label = {
            label: records[example_id] for label, records in candidate_by_label.items()
        }
        ordered_values, labels_by_value = syndrome._candidate_pool(rows_by_label, candidate_labels)
        fallback = syndrome._fallback_prediction(
            rows_by_label=rows_by_label,
            fallback_label=fallback_label,
            target_label=target_label,
        )
        selections = {
            condition: linear_probe._select_by_predicted_signature(
                ordered_values=ordered_values,
                signature=residue_predictions[condition][index],
                moduli=config.moduli,
                fallback=fallback,
            )
            for condition in prediction_conditions
        }
        selections["target_only"] = fallback
        selections["slots_only"] = syndrome._select_slots_only(
            ordered_values=ordered_values,
            labels_by_value=labels_by_value,
            fallback=fallback,
        )
        gold = syndrome._gold_numeric(target_by_id[example_id])
        rows.append(
            {
                "index": index,
                "example_id": example_id,
                "labels": [
                    label for label, ids in target_ids.items() if example_id in ids
                ],
                "gold_answer": gold,
                "teacher_prediction": syndrome._prediction_numeric(teacher_by_id[example_id]),
                "fallback_prediction": fallback,
                "candidate_pool_size": len(ordered_values),
                "candidate_pool_values": list(ordered_values),
                "candidate_pool_contains_gold": gold in set(ordered_values),
                "candidate_labels_for_gold": sorted(set(labels_by_value.get(gold, []))),
                "predicted_residue_signatures": {
                    condition: list(residue_predictions[condition][index])
                    for condition in prediction_conditions
                },
                "conditions": {
                    condition: {
                        "prediction": prediction,
                        "correct": prediction == gold,
                        "candidate_labels": sorted(
                            set(labels_by_value.get(str(prediction), []))
                        )
                        if prediction is not None
                        else [],
                    }
                    for condition, prediction in selections.items()
                },
            }
        )
    clean_ids = target_ids["clean_residual_targets"]
    target_self_ids = target_ids["target_self_repair"]
    teacher_only_ids = target_ids["teacher_only"]
    conditions = prediction_conditions + ("target_only", "slots_only")
    condition_summaries = {
        condition: _summarize_condition(
            rows,
            condition=condition,
            clean_ids=clean_ids,
            target_self_ids=target_self_ids,
            teacher_only_ids=teacher_only_ids,
        )
        for condition in conditions
    }
    control_conditions = tuple(condition for condition in conditions if condition != "matched")
    control_clean_union = set().union(
        *[set(condition_summaries[condition]["clean_correct_ids"]) for condition in control_conditions]
    )
    matched_clean = set(condition_summaries["matched"]["clean_correct_ids"])
    source_necessary_clean = matched_clean - control_clean_union
    criteria = {
        "min_correct": condition_summaries["matched"]["correct_count"] >= config.min_correct,
        "preserve_fallback_floor": (
            condition_summaries["matched"]["correct_count"]
            >= condition_summaries["target_only"]["correct_count"]
        ),
        "min_clean_source_necessary": (
            len(source_necessary_clean) >= config.min_clean_source_necessary
        ),
        "control_clean_union_empty": len(control_clean_union) == 0,
    }
    failing = [name for name, passed in criteria.items() if not passed]
    return {
        "moduli": list(config.moduli),
        "status": (
            "learned_syndrome_probe_clears_gate"
            if not failing
            else "learned_syndrome_probe_fails_gate"
        ),
        "criteria": criteria,
        "failing_criteria": failing,
        "condition_summaries": condition_summaries,
        "control_clean_union_ids": sorted(control_clean_union),
        "control_conditions": list(control_conditions),
        "source_necessary_clean_ids": sorted(source_necessary_clean),
        "candidate_pool_gold_count": sum(
            int(bool(row["candidate_pool_contains_gold"])) for row in rows
        ),
        "candidate_pool_clean_gold_count": sum(
            int(bool(row["candidate_pool_contains_gold"]))
            for row in rows
            if row["example_id"] in clean_ids
        ),
        "rows": rows,
    }


def analyze_with_token_features(
    *,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    feature_metadata: Sequence[dict[str, Any]],
    target_spec: syndrome.RowSpec,
    teacher_spec: syndrome.RowSpec,
    candidate_specs: Sequence[syndrome.RowSpec],
    target_set_path: pathlib.Path,
    fallback_label: str,
    config: LearnedProbeConfig,
    min_numeric_coverage: int,
    run_date: str,
    device: str,
) -> dict[str, Any]:
    target_records = syndrome._records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("target rows contain duplicate example_id values")
    feature_ids = [str(row["example_id"]) for row in feature_metadata]
    if feature_ids != reference_ids:
        raise ValueError("feature metadata IDs do not match target ordered IDs")
    teacher_records = syndrome._subset_reference_order(
        syndrome._records_for_method(teacher_spec),
        reference_ids,
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
        target_spec.label: target_by_id,
    }
    for spec in candidate_specs:
        if spec.label in candidate_by_label:
            raise ValueError(f"Duplicate candidate label {spec.label!r}")
        candidate_by_label[spec.label] = syndrome._by_id(
            syndrome._subset_reference_order(
                syndrome._records_for_method(spec),
                reference_ids,
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

    labels_by_modulus = linear_probe._teacher_residue_labels(teacher_records, config.moduli)
    residue_predictions, fold_metadata = _crossfit_residue_predictions(
        tokens=tokens.float().cpu(),
        mask=mask.cpu(),
        labels_by_modulus=labels_by_modulus,
        config=config,
        device=device,
    )
    run = _evaluate_learned_probe(
        reference_ids=reference_ids,
        target_by_id=target_by_id,
        teacher_by_id=teacher_by_id,
        candidate_by_label=candidate_by_label,
        target_label=target_spec.label,
        fallback_label=fallback_label,
        target_ids=target_ids,
        residue_predictions=residue_predictions,
        config=config,
    )
    status = run["status"] if not provenance_issues else "learned_syndrome_probe_fails_gate"
    return {
        "date": run_date,
        "status": status,
        "interpretation": (
            "This probe trains a cross-fitted tiny query bottleneck over frozen "
            "source token states to predict C2C residue classes. It is a "
            "strict source-syndrome diagnostic, not a generation method."
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
        },
        "reference_n": len(reference_ids),
        "target_ids": {key: sorted(value) for key, value in target_ids.items()},
        "feature_metadata": list(feature_metadata),
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
        "# SVAMP32 Learned Syndrome Probe",
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
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target", required=True, type=syndrome._parse_spec)
    parser.add_argument("--teacher", required=True, type=syndrome._parse_spec)
    parser.add_argument("--candidate", action="append", type=syndrome._parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target_self_repair")
    parser.add_argument("--moduli", default="2,3,5,7")
    parser.add_argument("--query-count", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--outer-folds", default="8")
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--min-correct", type=int, default=14)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--min-numeric-coverage", type=int, default=31)
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", choices=["true", "false"], default=None)
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
    config = LearnedProbeConfig(
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
    tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.source_model,
            torch_dtype=linear_probe._torch_dtype(args.dtype),
            trust_remote_code=True,
        )
        .to(args.device)
        .eval()
    )
    tokens, mask, feature_metadata = _extract_source_token_features(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=args.device,
        source_reasoning_mode=args.source_reasoning_mode,
        use_chat_template=bool(args.source_use_chat_template),
        enable_thinking=_bool_arg(args.source_enable_thinking),
        feature_layers=str(args.feature_layers),
    )
    payload = analyze_with_token_features(
        tokens=tokens,
        mask=mask,
        feature_metadata=feature_metadata,
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
