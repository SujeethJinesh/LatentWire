#!/usr/bin/env python3
"""Train a tiny source-latent syndrome probe on frozen SVAMP32 rows.

This is still a diagnostic, not a final method. It replaces the previous C2C
oracle syndrome with leave-one-out predictions from frozen source hidden-state
summaries, then evaluates the same target candidate pools and source-destroying
controls.
"""

from __future__ import annotations

import argparse
import math
import json
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
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


@dataclass(frozen=True)
class ProbeConfig:
    moduli: tuple[int, ...]
    probe_model: str = "ridge"
    ridge_lambda: float = 1.0
    shuffle_offset: int = 1
    min_correct: int = 14
    min_clean_source_necessary: int = 2
    query_slots: int = 8
    query_epochs: int = 80
    query_lr: float = 1e-2
    query_weight_decay: float = 1e-3
    query_seed: int = 0


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _torch_dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _ones(length: int, device: str) -> torch.Tensor:
    return torch.ones((1, length), device=device, dtype=torch.long)


def _feature_layer_indices(num_hidden_states: int, spec: str) -> list[int]:
    if num_hidden_states <= 0:
        raise ValueError("model returned no hidden states")
    if spec == "last":
        return [num_hidden_states - 1]
    if spec == "mid,last":
        return [max(0, num_hidden_states // 2), num_hidden_states - 1]
    if spec == "all":
        return list(range(num_hidden_states))
    indices: list[int] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0:
            idx = num_hidden_states + idx
        if idx < 0 or idx >= num_hidden_states:
            raise ValueError(f"feature layer index {raw!r} out of range 0..{num_hidden_states - 1}")
        indices.append(idx)
    if not indices:
        raise ValueError("empty feature layer spec")
    return indices


@torch.no_grad()
def _extract_source_features(
    *,
    model,
    tokenizer,
    examples,
    device: str,
    source_reasoning_mode: str,
    use_chat_template: bool,
    enable_thinking: bool | None,
    feature_layers: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
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
            attention_mask=_ones(input_ids.shape[1], device),
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = tuple(out.hidden_states)
        layer_indices = _feature_layer_indices(len(hidden_states), feature_layers)
        parts: list[torch.Tensor] = []
        for layer_idx in layer_indices:
            hidden = hidden_states[layer_idx][0].detach().float().cpu()
            parts.append(hidden[-1, :])
            parts.append(hidden.mean(dim=0))
        feature = torch.cat(parts, dim=0)
        rows.append(feature)
        metadata.append(
            {
                "example_id": _generation_example_id(example),
                "formatted_prompt_tokens": int(input_ids.shape[1]),
                "feature_dim": int(feature.numel()),
                "feature_layers": layer_indices,
            }
        )
    return torch.stack(rows, dim=0), metadata


def _teacher_residue_labels(
    teacher_rows: Sequence[dict[str, Any]],
    moduli: Sequence[int],
) -> dict[int, torch.Tensor]:
    labels: dict[int, list[int]] = {modulus: [] for modulus in moduli}
    for row in teacher_rows:
        numeric = syndrome._prediction_numeric(row)
        integer = syndrome._integer_value(numeric)
        if integer is None:
            raise ValueError(f"Teacher row lacks integer numeric prediction: {row.get('example_id')}")
        for modulus in moduli:
            labels[modulus].append(integer % modulus)
    return {
        modulus: torch.tensor(values, dtype=torch.long)
        for modulus, values in labels.items()
    }


def _standardize_train_test(
    train_x: torch.Tensor,
    test_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    test_was_1d = test_x.dim() == 1
    if test_was_1d:
        test_x = test_x.unsqueeze(0)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    scaled_test = (test_x - mean) / std
    if test_was_1d:
        scaled_test = scaled_test.squeeze(0)
    return (train_x - mean) / std, scaled_test


def _fit_ridge_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    num_classes: int,
    ridge_lambda: float,
) -> torch.Tensor:
    if train_x.shape[1] > train_x.shape[0]:
        # Equivalent to an unregularized-intercept ridge fit, but solves the
        # n x n dual system instead of a huge feature_dim x feature_dim system.
        y = torch.nn.functional.one_hot(train_y, num_classes=num_classes).float()
        mean_x = train_x.mean(dim=0, keepdim=True)
        mean_y = y.mean(dim=0, keepdim=True)
        x_centered = train_x - mean_x
        y_centered = y - mean_y
        gram = x_centered @ x_centered.T
        reg = torch.eye(gram.shape[0], dtype=train_x.dtype) * float(ridge_lambda)
        alpha = torch.linalg.solve(gram + reg, y_centered)
        weights = x_centered.T @ alpha
        bias = mean_y - mean_x @ weights
        return torch.cat([weights, bias], dim=0)

    x = torch.cat([train_x, torch.ones((train_x.shape[0], 1), dtype=train_x.dtype)], dim=1)
    y = torch.nn.functional.one_hot(train_y, num_classes=num_classes).float()
    xtx = x.T @ x
    reg = torch.eye(xtx.shape[0], dtype=x.dtype) * float(ridge_lambda)
    reg[-1, -1] = 0.0
    return torch.linalg.solve(xtx + reg, x.T @ y)


def _predict_with_weights(test_x: torch.Tensor, weights: torch.Tensor) -> int:
    x = torch.cat([test_x, torch.ones((1,), dtype=test_x.dtype)])
    scores = x @ weights
    return int(torch.argmax(scores).item())


class _QueryResidueProbe(torch.nn.Module):
    def __init__(self, hidden_dim: int, slots: int, moduli: Sequence[int]) -> None:
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(slots, hidden_dim) * 0.02)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.key = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.heads = torch.nn.ModuleDict(
            {str(modulus): torch.nn.Linear(slots * hidden_dim, int(modulus)) for modulus in moduli}
        )

    def forward(self, tokens: torch.Tensor) -> dict[int, torch.Tensor]:
        normalized = self.norm(tokens)
        key = self.key(normalized)
        value = self.value(normalized)
        scores = torch.einsum("qh,bth->bqt", self.query, key) / math.sqrt(key.shape[-1])
        pooled = torch.softmax(scores, dim=-1) @ value
        flat = pooled.flatten(start_dim=1)
        return {int(modulus): head(flat) for modulus, head in self.heads.items()}


def _feature_summary_tokens(
    features: torch.Tensor,
    feature_metadata: Sequence[dict[str, Any]],
) -> torch.Tensor:
    if not feature_metadata:
        raise ValueError("feature metadata is empty")
    token_count = len(feature_metadata[0]["feature_layers"]) * 2
    if token_count <= 0 or features.shape[1] % token_count != 0:
        raise ValueError("feature dimension cannot be reshaped into summary tokens")
    hidden_dim = features.shape[1] // token_count
    return features.reshape(features.shape[0], token_count, hidden_dim)


def _standardize_token_views(
    train_tokens: torch.Tensor,
    *eval_tokens: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    mean = train_tokens.mean(dim=(0, 1), keepdim=True)
    std = train_tokens.std(dim=(0, 1), unbiased=False, keepdim=True).clamp_min(1e-6)
    return tuple((tokens - mean) / std for tokens in (train_tokens, *eval_tokens))


def _train_query_residue_probe(
    train_tokens: torch.Tensor,
    labels_by_modulus: dict[int, torch.Tensor],
    *,
    config: ProbeConfig,
    seed: int,
) -> _QueryResidueProbe:
    torch.manual_seed(int(seed))
    model = _QueryResidueProbe(
        hidden_dim=int(train_tokens.shape[-1]),
        slots=int(config.query_slots),
        moduli=config.moduli,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.query_lr),
        weight_decay=float(config.query_weight_decay),
    )
    for _ in range(int(config.query_epochs)):
        optimizer.zero_grad()
        logits = model(train_tokens)
        loss = sum(
            torch.nn.functional.cross_entropy(logits[int(modulus)], labels)
            for modulus, labels in labels_by_modulus.items()
        )
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.no_grad()
def _predict_query_signature(model: _QueryResidueProbe, tokens: torch.Tensor, moduli: Sequence[int]) -> tuple[int, ...]:
    logits = model(tokens.unsqueeze(0))
    return tuple(int(torch.argmax(logits[int(modulus)], dim=-1).item()) for modulus in moduli)


def _loocv_query_bottleneck_predictions(
    features: torch.Tensor,
    feature_metadata: Sequence[dict[str, Any]],
    labels_by_modulus: dict[int, torch.Tensor],
    *,
    config: ProbeConfig,
) -> dict[str, list[tuple[int, ...]]]:
    tokens = _feature_summary_tokens(features, feature_metadata).float().cpu()
    n = int(tokens.shape[0])
    matched: list[tuple[int, ...]] = []
    zero_source: list[tuple[int, ...]] = []
    shuffled_source: list[tuple[int, ...]] = []
    label_shuffled: list[tuple[int, ...]] = []
    for idx in range(n):
        train_indices = [j for j in range(n) if j != idx]
        train_tokens_raw = tokens[train_indices]
        test_tokens_raw = tokens[idx : idx + 1]
        zero_tokens_raw = torch.zeros_like(test_tokens_raw)
        shuffled_idx = (idx + int(config.shuffle_offset)) % n
        if shuffled_idx == idx:
            shuffled_idx = (idx + 1) % n
        shuffled_tokens_raw = tokens[shuffled_idx : shuffled_idx + 1]
        train_tokens, test_tokens, zero_tokens, shuffled_tokens = _standardize_token_views(
            train_tokens_raw,
            test_tokens_raw,
            zero_tokens_raw,
            shuffled_tokens_raw,
        )
        train_labels = {
            modulus: labels_by_modulus[modulus][train_indices]
            for modulus in config.moduli
        }
        model = _train_query_residue_probe(
            train_tokens,
            train_labels,
            config=config,
            seed=int(config.query_seed) + idx,
        )
        matched.append(_predict_query_signature(model, test_tokens[0], config.moduli))
        zero_source.append(_predict_query_signature(model, zero_tokens[0], config.moduli))
        shuffled_source.append(_predict_query_signature(model, shuffled_tokens[0], config.moduli))

        shuffled_labels = {
            modulus: labels[torch.arange(labels.numel() - 1, -1, -1)]
            for modulus, labels in train_labels.items()
        }
        label_model = _train_query_residue_probe(
            train_tokens,
            shuffled_labels,
            config=config,
            seed=int(config.query_seed) + 10000 + idx,
        )
        label_shuffled.append(_predict_query_signature(label_model, test_tokens[0], config.moduli))
    return {
        "matched": matched,
        "zero_source": zero_source,
        "shuffled_source": shuffled_source,
        "label_shuffled": label_shuffled,
    }


def _loocv_residue_predictions(
    features: torch.Tensor,
    feature_metadata: Sequence[dict[str, Any]],
    labels_by_modulus: dict[int, torch.Tensor],
    *,
    config: ProbeConfig,
) -> dict[str, list[tuple[int, ...]]]:
    if config.probe_model == "query_bottleneck":
        return _loocv_query_bottleneck_predictions(
            features,
            feature_metadata,
            labels_by_modulus,
            config=config,
        )
    if config.probe_model != "ridge":
        raise ValueError(f"Unsupported probe model: {config.probe_model!r}")

    n = int(features.shape[0])
    matched: list[tuple[int, ...]] = []
    zero_source: list[tuple[int, ...]] = []
    shuffled_source: list[tuple[int, ...]] = []
    label_shuffled: list[tuple[int, ...]] = []
    for idx in range(n):
        train_indices = [j for j in range(n) if j != idx]
        train_x_raw = features[train_indices]
        test_x_raw = features[idx]
        zero_x_raw = torch.zeros_like(test_x_raw)
        shuffled_idx = (idx + int(config.shuffle_offset)) % n
        if shuffled_idx == idx:
            shuffled_idx = (idx + 1) % n
        shuffled_x_raw = features[shuffled_idx]
        matched_parts: list[int] = []
        zero_parts: list[int] = []
        shuffled_parts: list[int] = []
        label_shuffled_parts: list[int] = []
        for modulus in config.moduli:
            train_y = labels_by_modulus[modulus][train_indices]
            train_x, test_x = _standardize_train_test(train_x_raw, test_x_raw)
            _, zero_x = _standardize_train_test(train_x_raw, zero_x_raw)
            _, shuffled_x = _standardize_train_test(train_x_raw, shuffled_x_raw)
            weights = _fit_ridge_classifier(
                train_x,
                train_y,
                num_classes=int(modulus),
                ridge_lambda=float(config.ridge_lambda),
            )
            matched_parts.append(_predict_with_weights(test_x, weights))
            zero_parts.append(_predict_with_weights(zero_x, weights))
            shuffled_parts.append(_predict_with_weights(shuffled_x, weights))

            shuffled_y = train_y[torch.arange(train_y.numel() - 1, -1, -1)]
            label_weights = _fit_ridge_classifier(
                train_x,
                shuffled_y,
                num_classes=int(modulus),
                ridge_lambda=float(config.ridge_lambda),
            )
            label_shuffled_parts.append(_predict_with_weights(test_x, label_weights))
        matched.append(tuple(matched_parts))
        zero_source.append(tuple(zero_parts))
        shuffled_source.append(tuple(shuffled_parts))
        label_shuffled.append(tuple(label_shuffled_parts))
    return {
        "matched": matched,
        "zero_source": zero_source,
        "shuffled_source": shuffled_source,
        "label_shuffled": label_shuffled,
    }


def _select_by_predicted_signature(
    *,
    ordered_values: Sequence[str],
    signature: tuple[int, ...] | None,
    moduli: Sequence[int],
    fallback: str | None,
) -> str | None:
    if signature is None:
        return fallback
    for value in ordered_values:
        if syndrome._signature(value, moduli) == signature:
            return value
    return fallback


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


def _evaluate_source_latent_probe(
    *,
    reference_ids: Sequence[str],
    target_by_id: dict[str, dict[str, Any]],
    teacher_by_id: dict[str, dict[str, Any]],
    candidate_by_label: dict[str, dict[str, dict[str, Any]]],
    target_label: str,
    fallback_label: str,
    target_ids: dict[str, set[str]],
    residue_predictions: dict[str, list[tuple[int, ...]]],
    config: ProbeConfig,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    candidate_labels = list(candidate_by_label)
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
        gold = syndrome._gold_numeric(target_by_id[example_id])
        selections = {
            "matched": _select_by_predicted_signature(
                ordered_values=ordered_values,
                signature=residue_predictions["matched"][index],
                moduli=config.moduli,
                fallback=fallback,
            ),
            "zero_source": _select_by_predicted_signature(
                ordered_values=ordered_values,
                signature=residue_predictions["zero_source"][index],
                moduli=config.moduli,
                fallback=fallback,
            ),
            "shuffled_source": _select_by_predicted_signature(
                ordered_values=ordered_values,
                signature=residue_predictions["shuffled_source"][index],
                moduli=config.moduli,
                fallback=fallback,
            ),
            "label_shuffled": _select_by_predicted_signature(
                ordered_values=ordered_values,
                signature=residue_predictions["label_shuffled"][index],
                moduli=config.moduli,
                fallback=fallback,
            ),
            "target_only": fallback,
            "slots_only": syndrome._select_slots_only(
                ordered_values=ordered_values,
                labels_by_value=labels_by_value,
                fallback=fallback,
            ),
        }
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
                    condition: list(signature)
                    for condition, signatures in residue_predictions.items()
                    for signature in [signatures[index]]
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
    conditions = (
        "matched",
        "zero_source",
        "shuffled_source",
        "label_shuffled",
        "target_only",
        "slots_only",
    )
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
    control_clean_union = set().union(
        *[
            set(condition_summaries[condition]["clean_correct_ids"])
            for condition in (
                "zero_source",
                "shuffled_source",
                "label_shuffled",
                "target_only",
                "slots_only",
            )
        ]
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
    status = (
        "source_latent_syndrome_probe_clears_gate"
        if not failing
        else "source_latent_syndrome_probe_fails_gate"
    )
    return {
        "moduli": list(config.moduli),
        "ridge_lambda": float(config.ridge_lambda),
        "status": status,
        "criteria": criteria,
        "failing_criteria": failing,
        "condition_summaries": condition_summaries,
        "control_clean_union_ids": sorted(control_clean_union),
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


def analyze_with_features(
    *,
    features: torch.Tensor,
    feature_metadata: Sequence[dict[str, Any]],
    target_spec: syndrome.RowSpec,
    teacher_spec: syndrome.RowSpec,
    candidate_specs: Sequence[syndrome.RowSpec],
    target_set_path: pathlib.Path,
    fallback_label: str,
    config: ProbeConfig,
    min_numeric_coverage: int,
    run_date: str,
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

    labels_by_modulus = _teacher_residue_labels(teacher_records, config.moduli)
    residue_predictions = _loocv_residue_predictions(
        features.float().cpu(),
        feature_metadata,
        labels_by_modulus,
        config=config,
    )
    run = _evaluate_source_latent_probe(
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
    status = (
        run["status"]
        if not provenance_issues
        else "source_latent_syndrome_probe_fails_gate"
    )
    if config.probe_model == "query_bottleneck":
        interpretation = (
            "This probe trains leave-one-out learned query bottlenecks over "
            "source hidden summary tokens to predict C2C residue classes. It "
            "tests source-latent predictability of the previously cleared "
            "syndrome bound, but remains a small-slice diagnostic."
        )
    else:
        interpretation = (
            "This probe trains leave-one-out ridge classifiers from frozen "
            "source hidden summaries to C2C residue classes. It tests "
            "source-latent predictability of the previously cleared syndrome "
            "bound, but remains a small-slice diagnostic."
        )
    return {
        "date": run_date,
        "status": status,
        "interpretation": interpretation,
        "artifacts": {
            "target": {
                "label": target_spec.label,
                "path": _display_path(target_spec.path),
                "method": target_spec.method,
            },
            "teacher": {
                "label": teacher_spec.label,
                "path": _display_path(teacher_spec.path),
                "method": teacher_spec.method,
            },
            "target_set_json": _display_path(target_set_path),
            "candidates": [
                {
                    "label": spec.label,
                    "path": _display_path(spec.path),
                    "method": spec.method,
                }
                for spec in candidate_specs
            ],
        },
        "config": {
            "fallback_label": fallback_label,
            "probe_model": config.probe_model,
            "moduli": list(config.moduli),
            "ridge_lambda": float(config.ridge_lambda),
            "shuffle_offset": int(config.shuffle_offset),
            "min_correct": int(config.min_correct),
            "min_clean_source_necessary": int(config.min_clean_source_necessary),
            "min_numeric_coverage": int(min_numeric_coverage),
            "query_slots": int(config.query_slots),
            "query_epochs": int(config.query_epochs),
            "query_lr": float(config.query_lr),
            "query_weight_decay": float(config.query_weight_decay),
            "query_seed": int(config.query_seed),
        },
        "reference_n": len(reference_ids),
        "target_ids": {key: sorted(value) for key, value in target_ids.items()},
        "feature_metadata": list(feature_metadata),
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
    matched = run["condition_summaries"]["matched"]
    target_only = run["condition_summaries"]["target_only"]
    zero_source = run["condition_summaries"]["zero_source"]
    shuffled_source = run["condition_summaries"]["shuffled_source"]
    label_shuffled = run["condition_summaries"]["label_shuffled"]
    slots_only = run["condition_summaries"]["slots_only"]
    lines = [
        "# SVAMP32 Source-Latent Syndrome Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- moduli: `{','.join(str(value) for value in payload['config']['moduli'])}`",
        f"- probe model: `{payload['config']['probe_model']}`",
        f"- ridge lambda: `{payload['config']['ridge_lambda']}`",
        f"- teacher numeric coverage: `{payload['provenance']['teacher_numeric_coverage']}/{payload['reference_n']}`",
        f"- provenance issues: `{len(payload['provenance']['issues'])}`",
    ]
    if payload["config"]["probe_model"] == "query_bottleneck":
        lines.extend(
            [
                f"- query slots: `{payload['config']['query_slots']}`",
                f"- query epochs: `{payload['config']['query_epochs']}`",
                f"- query lr: `{payload['config']['query_lr']}`",
                f"- query weight decay: `{payload['config']['query_weight_decay']}`",
                f"- query seed: `{payload['config']['query_seed']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Condition | Correct | Clean Correct | Target-Self Correct |",
            "|---|---:|---:|---:|",
        ]
    )
    for summary in (
        matched,
        zero_source,
        shuffled_source,
        label_shuffled,
        target_only,
        slots_only,
    ):
        lines.append(
            "| {condition} | {correct} | {clean} | {target_self} |".format(
                condition=summary["condition"],
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
    parser.add_argument("--probe-model", choices=["ridge", "query_bottleneck"], default="ridge")
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--min-correct", type=int, default=14)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--min-numeric-coverage", type=int, default=31)
    parser.add_argument("--query-slots", type=int, default=8)
    parser.add_argument("--query-epochs", type=int, default=80)
    parser.add_argument("--query-lr", type=float, default=1e-2)
    parser.add_argument("--query-weight-decay", type=float, default=1e-3)
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--feature-layers", default="last")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    examples = load_generation(str(_resolve(args.eval_file)))
    moduli = tuple(int(value) for value in str(args.moduli).split(",") if value.strip())
    config = ProbeConfig(
        moduli=moduli,
        probe_model=str(args.probe_model),
        ridge_lambda=float(args.ridge_lambda),
        shuffle_offset=int(args.shuffle_offset),
        min_correct=int(args.min_correct),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
        query_slots=int(args.query_slots),
        query_epochs=int(args.query_epochs),
        query_lr=float(args.query_lr),
        query_weight_decay=float(args.query_weight_decay),
        query_seed=int(args.query_seed),
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading source model: {args.source_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.source_model,
            torch_dtype=_torch_dtype(args.dtype),
            trust_remote_code=True,
        )
        .to(args.device)
        .eval()
    )
    features, feature_metadata = _extract_source_features(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        device=args.device,
        source_reasoning_mode=args.source_reasoning_mode,
        use_chat_template=bool(args.source_use_chat_template),
        enable_thinking=_bool_arg(args.source_enable_thinking),
        feature_layers=str(args.feature_layers),
    )
    payload = analyze_with_features(
        features=features,
        feature_metadata=feature_metadata,
        target_spec=args.target,
        teacher_spec=args.teacher,
        candidate_specs=args.candidate,
        target_set_path=_resolve(args.target_set_json),
        fallback_label=str(args.fallback_label),
        config=config,
        min_numeric_coverage=int(args.min_numeric_coverage),
        run_date=str(args.date),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_md, payload)
    print(
        json.dumps(
            {"status": payload["status"], "output_json": _display_path(output_json)},
            indent=2,
        ),
        flush=True,
    )
    return payload


if __name__ == "__main__":
    main()
