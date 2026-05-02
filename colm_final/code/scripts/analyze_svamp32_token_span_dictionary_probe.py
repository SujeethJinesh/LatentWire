#!/usr/bin/env python3
"""Fold-local token/span dictionary probe for SVAMP32 C2C headroom."""

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
from scripts import analyze_svamp32_learned_syndrome_probe as token_probe
from scripts import analyze_svamp32_source_latent_syndrome_probe as latent_probe
from scripts import analyze_svamp32_sparse_anchor_sidecar_probe as sparse_anchor
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


@dataclass(frozen=True)
class DictionaryProbeConfig:
    moduli: tuple[int, ...]
    outer_folds: str = "8"
    atoms: int = 32
    topk_atoms: int = 2
    random_projection_dim: int = 128
    dictionary_iters: int = 8
    ridge_lambda: float = 1.0
    shuffle_offset: int = 1
    min_correct: int = 10
    min_clean_source_necessary: int = 2
    max_control_clean_union: int = 0
    sequence_sidecar_dim: int = 32
    sequence_scale: float = 0.35
    profile_scale: float = 0.20
    token_scale: float = 0.75
    projection_seed: int = 0


def _resolve(path: str | pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _torch_dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _bool_arg(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.lower() == "true"


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


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


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


def _random_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn(
        (int(input_dim), int(output_dim)),
        generator=generator,
        dtype=torch.float32,
    ) / math.sqrt(max(int(input_dim), 1))


def _spherical_kmeans(
    vectors: torch.Tensor,
    *,
    atoms: int,
    iters: int,
) -> torch.Tensor:
    vectors = _normalize_rows(vectors.float())
    if vectors.shape[0] < atoms:
        repeats = math.ceil(atoms / max(int(vectors.shape[0]), 1))
        vectors = vectors.repeat((repeats, 1))
    init_idx = torch.linspace(0, vectors.shape[0] - 1, steps=atoms).round().long()
    centers = vectors[init_idx].clone()
    for _ in range(max(1, int(iters))):
        sims = vectors @ centers.T
        assignment = torch.argmax(sims, dim=1)
        next_centers = []
        for atom in range(atoms):
            members = vectors[assignment == atom]
            if members.numel() == 0:
                next_centers.append(centers[atom])
            else:
                next_centers.append(members.mean(dim=0))
        centers = _normalize_rows(torch.stack(next_centers, dim=0))
    return centers


def _encode_dictionary(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    topk_atoms: int,
) -> torch.Tensor:
    valid = _normalize_rows(tokens[mask].float())
    code = torch.zeros(dictionary.shape[0], dtype=torch.float32)
    if valid.numel() == 0:
        return code
    sims = valid @ dictionary.T
    values, indices = torch.topk(sims.clamp_min(0.0), k=max(1, min(topk_atoms, dictionary.shape[0])), dim=1)
    code.scatter_add_(0, indices.reshape(-1), values.reshape(-1))
    return code / code.norm().clamp_min(1e-8)


def _same_norm_noise(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(tokens.shape, generator=generator, dtype=tokens.dtype)
    valid_norm = tokens[mask].norm().clamp_min(1e-6)
    noise_norm = noise[mask].norm().clamp_min(1e-6)
    return noise * (valid_norm / noise_norm)


def _fit_signature_classifiers(
    train_x: torch.Tensor,
    labels_by_modulus: dict[int, torch.Tensor],
    train_indices: Sequence[int],
    *,
    config: DictionaryProbeConfig,
    label_shuffle: bool,
) -> dict[int, torch.Tensor]:
    weights: dict[int, torch.Tensor] = {}
    for modulus in config.moduli:
        train_y = labels_by_modulus[modulus][list(train_indices)]
        if label_shuffle:
            train_y = train_y[torch.arange(train_y.numel() - 1, -1, -1)]
        weights[modulus] = latent_probe._fit_ridge_classifier(
            train_x,
            train_y,
            num_classes=int(modulus),
            ridge_lambda=float(config.ridge_lambda),
        )
    return weights


def _predict_signature(
    x: torch.Tensor,
    weights_by_modulus: dict[int, torch.Tensor],
    moduli: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        latent_probe._predict_with_weights(x, weights_by_modulus[int(modulus)])
        for modulus in moduli
    )


def _standardize_vectors(
    train_x: torch.Tensor,
    eval_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std


def _crossfit_dictionary_predictions(
    *,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    sidecar: torch.Tensor,
    profile: torch.Tensor,
    labels_by_modulus: dict[int, torch.Tensor],
    config: DictionaryProbeConfig,
) -> tuple[dict[str, list[tuple[int, ...]]], list[dict[str, Any]]]:
    n = int(tokens.shape[0])
    folds = _make_outer_folds(n, config.outer_folds)
    predictions: dict[str, list[tuple[int, ...] | None]] = {
        "matched": [None] * n,
        "zero_source": [None] * n,
        "shuffled_source": [None] * n,
        "label_shuffled": [None] * n,
        "same_norm_noise": [None] * n,
        "boundary_only": [None] * n,
    }
    metadata: list[dict[str, Any]] = []
    projection = _random_projection(
        int(tokens.shape[-1]),
        int(config.random_projection_dim),
        int(config.projection_seed),
    )
    for fold_idx, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_indices = [idx for idx in range(n) if idx not in heldout_set]
        standardized = _standardize_tokens_for_fold(tokens, mask, train_indices)
        projected = standardized @ projection
        train_valid = projected[list(train_indices)][mask[list(train_indices)]]
        dictionary = _spherical_kmeans(
            train_valid,
            atoms=int(config.atoms),
            iters=int(config.dictionary_iters),
        )

        train_codes = torch.stack(
            [
                _encode_dictionary(
                    projected[idx],
                    mask[idx],
                    dictionary,
                    topk_atoms=int(config.topk_atoms),
                )
                for idx in train_indices
            ],
            dim=0,
        )
        train_joint_raw = torch.cat(
            [
                train_codes,
                float(config.sequence_scale) * sidecar[list(train_indices)],
                float(config.profile_scale) * profile[list(train_indices)],
            ],
            dim=1,
        )
        train_joint, _ = _standardize_vectors(train_joint_raw, train_joint_raw)
        matched_weights = _fit_signature_classifiers(
            train_joint,
            labels_by_modulus,
            train_indices,
            config=config,
            label_shuffle=False,
        )
        label_weights = _fit_signature_classifiers(
            train_joint,
            labels_by_modulus,
            train_indices,
            config=config,
            label_shuffle=True,
        )

        dead_atoms = int((train_codes.sum(dim=0) <= 1e-8).sum().item())
        active_mass = train_codes.sum(dim=0)
        probs = active_mass / active_mass.sum().clamp_min(1e-8)
        entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum()
        perplexity = float(torch.exp(entropy).item())
        metadata.append(
            {
                "fold": fold_idx,
                "heldout_indices": list(heldout),
                "train_count": len(train_indices),
                "dead_atom_rate": float(dead_atoms / max(int(config.atoms), 1)),
                "codebook_perplexity": perplexity,
            }
        )

        for idx in heldout:
            own_code = _encode_dictionary(
                projected[idx],
                mask[idx],
                dictionary,
                topk_atoms=int(config.topk_atoms),
            )
            shuffled_idx = (idx + int(config.shuffle_offset)) % n
            if shuffled_idx == idx:
                shuffled_idx = (idx + 1) % n
            shuffled_code = _encode_dictionary(
                projected[shuffled_idx],
                mask[shuffled_idx],
                dictionary,
                topk_atoms=int(config.topk_atoms),
            )
            gen = torch.Generator().manual_seed(71_111 + int(config.projection_seed) * 997 + idx)
            noise_tokens = _same_norm_noise(projected[idx], mask[idx], generator=gen)
            noise_code = _encode_dictionary(
                noise_tokens,
                mask[idx],
                dictionary,
                topk_atoms=int(config.topk_atoms),
            )
            zero_code = torch.zeros_like(own_code)
            eval_raw = torch.stack(
                [
                    torch.cat([own_code, float(config.sequence_scale) * sidecar[idx], float(config.profile_scale) * profile[idx]]),
                    torch.cat([zero_code, float(config.sequence_scale) * sidecar[idx], float(config.profile_scale) * profile[idx]]),
                    torch.cat([shuffled_code, float(config.sequence_scale) * sidecar[idx], float(config.profile_scale) * profile[idx]]),
                    torch.cat([noise_code, float(config.sequence_scale) * sidecar[idx], float(config.profile_scale) * profile[idx]]),
                    torch.cat([zero_code, float(config.sequence_scale) * sidecar[idx], float(config.profile_scale) * profile[idx]]),
                ],
                dim=0,
            )
            _, eval_x = _standardize_vectors(train_joint_raw, eval_raw)
            predictions["matched"][idx] = _predict_signature(eval_x[0], matched_weights, config.moduli)
            predictions["zero_source"][idx] = _predict_signature(eval_x[1], matched_weights, config.moduli)
            predictions["shuffled_source"][idx] = _predict_signature(eval_x[2], matched_weights, config.moduli)
            predictions["same_norm_noise"][idx] = _predict_signature(eval_x[3], matched_weights, config.moduli)
            predictions["boundary_only"][idx] = _predict_signature(eval_x[4], matched_weights, config.moduli)
            predictions["label_shuffled"][idx] = _predict_signature(eval_x[0], label_weights, config.moduli)

    return {
        key: [value for value in values if value is not None]
        for key, values in predictions.items()
    }, metadata


def _select_by_signature(
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


def _evaluate(
    *,
    reference_ids: Sequence[str],
    target_by_id: dict[str, dict[str, Any]],
    teacher_by_id: dict[str, dict[str, Any]],
    candidate_by_label: dict[str, dict[str, dict[str, Any]]],
    target_label: str,
    fallback_label: str,
    target_ids: dict[str, set[str]],
    residue_predictions: dict[str, list[tuple[int, ...]]],
    config: DictionaryProbeConfig,
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
            condition: _select_by_signature(
                ordered_values=ordered_values,
                signature=residue_predictions[condition][index],
                moduli=config.moduli,
                fallback=fallback,
            )
            for condition in residue_predictions
        }
        selections["target_only"] = fallback
        selections["slots_only"] = syndrome._select_slots_only(
            ordered_values=ordered_values,
            labels_by_value=labels_by_value,
            fallback=fallback,
        )
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
                        "candidate_labels": sorted(set(labels_by_value.get(str(prediction), [])))
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
        "same_norm_noise",
        "boundary_only",
        "target_only",
        "slots_only",
    )
    summaries = {
        condition: _summarize_condition(
            rows,
            condition=condition,
            clean_ids=clean_ids,
            target_self_ids=target_self_ids,
            teacher_only_ids=teacher_only_ids,
        )
        for condition in conditions
    }
    control_conditions = (
        "zero_source",
        "shuffled_source",
        "label_shuffled",
        "same_norm_noise",
        "boundary_only",
        "target_only",
        "slots_only",
    )
    control_clean_union = set().union(
        *[set(summaries[condition]["clean_correct_ids"]) for condition in control_conditions]
    )
    matched_clean = set(summaries["matched"]["clean_correct_ids"])
    source_necessary = matched_clean - control_clean_union
    criteria = {
        "min_correct": summaries["matched"]["correct_count"] >= config.min_correct,
        "preserve_fallback_floor": (
            summaries["matched"]["correct_count"] >= summaries["target_only"]["correct_count"]
        ),
        "min_clean_source_necessary": len(source_necessary) >= config.min_clean_source_necessary,
        "max_control_clean_union": len(control_clean_union) <= config.max_control_clean_union,
    }
    failing = [name for name, passed in criteria.items() if not passed]
    return {
        "status": "token_span_dictionary_probe_clears_gate" if not failing else "token_span_dictionary_probe_fails_gate",
        "criteria": criteria,
        "failing_criteria": failing,
        "condition_summaries": summaries,
        "control_clean_union_ids": sorted(control_clean_union),
        "source_necessary_clean_ids": sorted(source_necessary),
        "candidate_pool_gold_count": sum(int(bool(row["candidate_pool_contains_gold"])) for row in rows),
        "candidate_pool_clean_gold_count": sum(
            int(bool(row["candidate_pool_contains_gold"]))
            for row in rows
            if row["example_id"] in clean_ids
        ),
        "rows": rows,
    }


def analyze_with_features(
    *,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    feature_metadata: Sequence[dict[str, Any]],
    sidecar: torch.Tensor,
    profile: torch.Tensor,
    sidecar_metadata: Sequence[dict[str, Any]],
    target_spec: syndrome.RowSpec,
    teacher_spec: syndrome.RowSpec,
    candidate_specs: Sequence[syndrome.RowSpec],
    target_set_path: pathlib.Path,
    fallback_label: str,
    config: DictionaryProbeConfig,
    min_numeric_coverage: int,
    run_date: str,
) -> dict[str, Any]:
    target_records = syndrome._records_for_method(target_spec)
    reference_ids = [str(row["example_id"]) for row in target_records]
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

    labels_by_modulus = latent_probe._teacher_residue_labels(teacher_records, config.moduli)
    residue_predictions, fold_metadata = _crossfit_dictionary_predictions(
        tokens=tokens.float().cpu(),
        mask=mask.cpu(),
        sidecar=sidecar.float().cpu(),
        profile=profile.float().cpu(),
        labels_by_modulus=labels_by_modulus,
        config=config,
    )
    run = _evaluate(
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
    status = run["status"] if not provenance_issues else "token_span_dictionary_probe_fails_gate"
    return {
        "date": run_date,
        "status": status,
        "interpretation": (
            "This probe learns a fold-local sparse dictionary over source token "
            "states, combines sparse token codes with source/target tokenizer "
            "boundary sidecar features, and decodes C2C residue signatures "
            "through the strict SVAMP32 candidate pool."
        ),
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
            "moduli": list(config.moduli),
            "outer_folds": config.outer_folds,
            "atoms": int(config.atoms),
            "topk_atoms": int(config.topk_atoms),
            "random_projection_dim": int(config.random_projection_dim),
            "dictionary_iters": int(config.dictionary_iters),
            "ridge_lambda": float(config.ridge_lambda),
            "shuffle_offset": int(config.shuffle_offset),
            "min_correct": int(config.min_correct),
            "min_clean_source_necessary": int(config.min_clean_source_necessary),
            "max_control_clean_union": int(config.max_control_clean_union),
            "sequence_sidecar_dim": int(config.sequence_sidecar_dim),
            "sequence_scale": float(config.sequence_scale),
            "profile_scale": float(config.profile_scale),
            "token_scale": float(config.token_scale),
            "projection_seed": int(config.projection_seed),
            "estimated_sidecar_bytes": int(config.topk_atoms) * 6 + int(config.sequence_sidecar_dim) // 8 + 6,
        },
        "reference_n": len(reference_ids),
        "target_ids": {key: sorted(value) for key, value in target_ids.items()},
        "feature_metadata": [
            {**dict(feature), **dict(side)}
            for feature, side in zip(feature_metadata, sidecar_metadata, strict=True)
        ],
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
        "# SVAMP32 Token/Span Dictionary Probe",
        "",
        f"- date: `{payload['date']}`",
        f"- status: `{payload['status']}`",
        f"- reference rows: `{payload['reference_n']}`",
        f"- moduli: `{','.join(str(value) for value in payload['config']['moduli'])}`",
        f"- atoms: `{payload['config']['atoms']}`",
        f"- top-k atoms: `{payload['config']['topk_atoms']}`",
        f"- random projection dim: `{payload['config']['random_projection_dim']}`",
        f"- estimated sidecar bytes: `{payload['config']['estimated_sidecar_bytes']}`",
        f"- teacher numeric coverage: `{payload['provenance']['teacher_numeric_coverage']}/{payload['reference_n']}`",
        f"- provenance issues: `{len(payload['provenance']['issues'])}`",
        "",
        "## Summary",
        "",
        "| Condition | Correct | Clean Correct | Target-Self Correct |",
        "|---|---:|---:|---:|",
    ]
    for condition in (
        "matched",
        "zero_source",
        "shuffled_source",
        "label_shuffled",
        "same_norm_noise",
        "boundary_only",
        "target_only",
        "slots_only",
    ):
        summary = run["condition_summaries"][condition]
        lines.append(
            "| {condition} | {correct} | {clean} | {target_self} |".format(
                condition=condition,
                correct=summary["correct_count"],
                clean=summary["clean_correct_count"],
                target_self=summary["target_self_correct_count"],
            )
        )
    mean_dead = sum(float(row["dead_atom_rate"]) for row in payload["fold_metadata"]) / max(len(payload["fold_metadata"]), 1)
    mean_perplexity = sum(float(row["codebook_perplexity"]) for row in payload["fold_metadata"]) / max(len(payload["fold_metadata"]), 1)
    lines.extend(
        [
            "",
            f"- mean dead atom rate: `{mean_dead:.4f}`",
            f"- mean codebook perplexity: `{mean_perplexity:.4f}`",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-tokenizer-model", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--target", required=True, type=syndrome._parse_spec)
    parser.add_argument("--teacher", required=True, type=syndrome._parse_spec)
    parser.add_argument("--candidate", action="append", type=syndrome._parse_spec, default=[])
    parser.add_argument("--target-set-json", required=True)
    parser.add_argument("--fallback-label", default="target")
    parser.add_argument("--moduli", default="2,3,5,7")
    parser.add_argument("--outer-folds", default="8")
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--topk-atoms", type=int, default=2)
    parser.add_argument("--random-projection-dim", type=int, default=128)
    parser.add_argument("--dictionary-iters", type=int, default=8)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--shuffle-offset", type=int, default=1)
    parser.add_argument("--min-correct", type=int, default=10)
    parser.add_argument("--min-clean-source-necessary", type=int, default=2)
    parser.add_argument("--max-control-clean-union", type=int, default=0)
    parser.add_argument("--min-numeric-coverage", type=int, default=26)
    parser.add_argument("--source-reasoning-mode", default="brief_analysis")
    parser.add_argument("--source-use-chat-template", action="store_true")
    parser.add_argument("--source-enable-thinking", choices=["true", "false"], default=None)
    parser.add_argument("--feature-layers", default="mid,last")
    parser.add_argument("--sequence-sidecar-dim", type=int, default=32)
    parser.add_argument("--sequence-scale", type=float, default=0.35)
    parser.add_argument("--profile-scale", type=float, default=0.20)
    parser.add_argument("--token-scale", type=float, default=0.75)
    parser.add_argument("--projection-seed", type=int, default=0)
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading source model: {args.source_model}", flush=True)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=True)
    if source_tokenizer.pad_token_id is None:
        source_tokenizer.pad_token = source_tokenizer.eos_token
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer_model, trust_remote_code=True)
    if target_tokenizer.pad_token_id is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.source_model,
            torch_dtype=_torch_dtype(args.dtype),
            trust_remote_code=True,
        )
        .to(args.device)
        .eval()
    )
    tokens, mask, feature_metadata = token_probe._extract_source_token_features(
        model=model,
        tokenizer=source_tokenizer,
        examples=examples,
        device=str(args.device),
        source_reasoning_mode=str(args.source_reasoning_mode),
        use_chat_template=bool(args.source_use_chat_template),
        enable_thinking=_bool_arg(args.source_enable_thinking),
        feature_layers=str(args.feature_layers),
    )
    prompts = [str(example.prompt) for example in examples]
    sidecar, profile, sidecar_metadata = sparse_anchor._sequence_alignment_sidecar(
        prompts,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        dim=int(args.sequence_sidecar_dim),
        token_scale=float(args.token_scale),
    )
    config = DictionaryProbeConfig(
        moduli=moduli,
        outer_folds=str(args.outer_folds),
        atoms=int(args.atoms),
        topk_atoms=int(args.topk_atoms),
        random_projection_dim=int(args.random_projection_dim),
        dictionary_iters=int(args.dictionary_iters),
        ridge_lambda=float(args.ridge_lambda),
        shuffle_offset=int(args.shuffle_offset),
        min_correct=int(args.min_correct),
        min_clean_source_necessary=int(args.min_clean_source_necessary),
        max_control_clean_union=int(args.max_control_clean_union),
        sequence_sidecar_dim=int(args.sequence_sidecar_dim),
        sequence_scale=float(args.sequence_scale),
        profile_scale=float(args.profile_scale),
        token_scale=float(args.token_scale),
        projection_seed=int(args.projection_seed),
    )
    payload = analyze_with_features(
        tokens=tokens,
        mask=mask,
        feature_metadata=feature_metadata,
        sidecar=sidecar,
        profile=profile,
        sidecar_metadata=sidecar_metadata,
        target_spec=args.target,
        teacher_spec=args.teacher,
        candidate_specs=args.candidate,
        target_set_path=_resolve(args.target_set_json),
        fallback_label=str(args.fallback_label),
        config=config,
        min_numeric_coverage=int(args.min_numeric_coverage),
        run_date=str(args.date),
    )
    payload["artifacts"]["source_model"] = str(args.source_model)
    payload["artifacts"]["target_tokenizer_model"] = str(args.target_tokenizer_model)
    payload["config"]["feature_layers"] = str(args.feature_layers)

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
