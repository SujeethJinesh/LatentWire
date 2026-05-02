#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_toy_quotient_gpa_sparse_dictionary as core
from scripts import run_toy_quotient_gpa_sparse_dictionary_byte_sidecar as sidecar
from scripts import run_toy_quotient_gpa_sparse_dictionary_interface_stress as iface
from scripts import run_toy_tokenizer_frontier_bridge as tokenizer_bridge


METHODS: tuple[str, ...] = (
    "heldout_fewshot_ridge_token_id",
    "heldout_fewshot_ridge_byte_span_remap",
    "quotient_gpa_sparse_dictionary_token_id",
    "quotient_gpa_sparse_dictionary_byte_span_remap",
    "quotient_gpa_sparse_dictionary_byte_sidecar_token_id",
    "quotient_gpa_sparse_dictionary_byte_sidecar_remap",
    "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id",
    "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap",
    "quotient_gpa_sparse_dictionary_oracle_interface",
)


ROW_KEY_ORDER: tuple[str, ...] = (
    "shot",
    "method",
    "seed",
    "heldout_family",
    "accuracy",
    "mse",
    "accuracy_delta_vs_token_id_fewshot",
    "mse_delta_vs_token_id_fewshot",
    "centroid_cosine",
    "mean_boundary_f1",
    "mean_remap_coverage",
    "mean_interface_noise_scale",
    "mean_sidecar_norm",
    "mean_alignment_sidecar_norm",
    "head_match_accuracy",
    "atom_recovery",
    "dead_atom_rate",
    "codebook_perplexity",
    "shared_basis",
    "heldout_pairs_used",
)


@dataclass(frozen=True)
class ToyQuotientGPASparseDictionarySequenceAlignedSidecarConfig:
    seed: int = 0
    heads: int = 4
    head_dim: int = 6
    atoms: int = 10
    classes: int = 5
    families: int = 5
    heldout_family: int = 4
    anchor_family: int = 0
    seen_shots_per_class: int = 24
    heldout_shots: tuple[int, ...] = (1, 2, 4, 8)
    test_examples_per_class: int = 64
    class_noise: float = 0.16
    source_noise: float = 0.03
    nuisance_rank: int = 3
    nuisance_strength: float = 0.22
    head_scale_jitter: float = 0.18
    head_bias_scale: float = 0.18
    ridge_lam: float = 1e-2
    gpa_iters: int = 8
    dictionary_iters: int = 10
    topk_atoms: int = 2
    remap_capacity: int = 10
    interface_strength: float = 2.5
    remap_recovery: float = 0.70
    sidecar_hash_dim: int = 16
    sidecar_ngram_max: int = 2
    sidecar_scale: float = 0.35
    alignment_hash_dim: int = 16
    alignment_scale: float = 0.30
    alignment_profile_scale: float = 0.20
    alignment_token_scale: float = 0.75


def _base_core_config(
    config: ToyQuotientGPASparseDictionarySequenceAlignedSidecarConfig,
) -> core.ToyQuotientGPASparseDictionaryConfig:
    return core.ToyQuotientGPASparseDictionaryConfig(
        seed=config.seed,
        heads=config.heads,
        head_dim=config.head_dim,
        atoms=config.atoms,
        classes=config.classes,
        families=config.families,
        heldout_family=config.heldout_family,
        anchor_family=config.anchor_family,
        seen_shots_per_class=config.seen_shots_per_class,
        heldout_shots=config.heldout_shots,
        test_examples_per_class=config.test_examples_per_class,
        class_noise=config.class_noise,
        source_noise=config.source_noise,
        nuisance_rank=config.nuisance_rank,
        nuisance_strength=config.nuisance_strength,
        head_scale_jitter=config.head_scale_jitter,
        head_bias_scale=config.head_bias_scale,
        ridge_lam=config.ridge_lam,
        gpa_iters=config.gpa_iters,
        dictionary_iters=config.dictionary_iters,
        topk_atoms=config.topk_atoms,
    )


def _sequence_aligned_sidecar_features(
    texts: list[str],
    *,
    source_tokenizer: tokenizer_bridge.ToyTokenizer,
    target_tokenizer: tokenizer_bridge.ToyTokenizer,
    dim: int,
    token_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.zeros(len(texts), dim, dtype=torch.float32)
    profiles = torch.zeros(len(texts), 6, dtype=torch.float32)
    for row, text in enumerate(texts):
        source_tokens = source_tokenizer.segment(text)
        target_tokens = target_tokenizer.segment(text)
        source_bounds = tokenizer_bridge._boundary_positions(text, source_tokens)
        target_bounds = tokenizer_bridge._boundary_positions(text, target_tokens)
        union_bounds = sorted((source_bounds | target_bounds) | {len(text)})

        start = 0
        for end in union_bounds:
            span = text[start:end]
            if span:
                weight = 1.0 + 0.25 * float(end in source_bounds) + 0.25 * float(end in target_bounds)
                idx = sidecar._stable_hash(b"span:" + span.encode("utf-8")) % dim
                features[row, idx] += weight
            start = end

        for token in source_tokens:
            idx = sidecar._stable_hash(b"src:" + token.encode("utf-8")) % dim
            features[row, idx] += float(token_scale)
        for token in target_tokens:
            idx = sidecar._stable_hash(b"tgt:" + token.encode("utf-8")) % dim
            features[row, idx] += float(token_scale)

        boundary_agreement = tokenizer_bridge._boundary_f1(source_bounds, target_bounds)
        source_len = float(len(source_tokens))
        target_len = float(len(target_tokens))
        fragmentation_gap = abs(source_len - target_len) / max(source_len + target_len, 1.0)
        agreement_gap = 1.0 - float(boundary_agreement)
        profiles[row] = torch.tensor(
            [
                boundary_agreement,
                agreement_gap,
                source_len / max(len(text), 1),
                target_len / max(len(text), 1),
                fragmentation_gap,
                len(union_bounds) / max(len(text), 1),
            ],
            dtype=torch.float32,
        )

        feature_norm = features[row].norm().clamp_min(1e-8)
        features[row] = features[row] / feature_norm
    profiles = profiles / profiles.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return features, profiles


def _compose_pipeline_with_sequence_aligned_sidecar(
    config: ToyQuotientGPASparseDictionarySequenceAlignedSidecarConfig,
    problem: dict[str, torch.Tensor],
    *,
    seen_norm_items: list[dict[str, Any]],
    heldout_fewshot_norm: dict[str, Any],
    heldout_test_norm: dict[str, Any],
    source_tokenizer: tokenizer_bridge.ToyTokenizer,
    target_tokenizer: tokenizer_bridge.ToyTokenizer,
) -> tuple[torch.Tensor, float, float, float, float, float, float]:
    anchor_item = next(item for item in seen_norm_items if int(item["family"][0].item()) == config.anchor_family)
    anchor_centroids = core._centroids_by_head(anchor_item["source"], anchor_item["label"], config.classes, config.heads, config.head_dim)

    quotient_seen_inputs = []
    quotient_seen_latent = []
    quotient_seen_centroids = []
    quotient_seen_centroid_means = []
    seen_texts: list[str] = []
    for item in seen_norm_items:
        centroids = core._centroids_by_head(item["source"], item["label"], config.classes, config.heads, config.head_dim)
        assignment, transforms = core._match_family_heads(centroids, anchor_centroids)
        quotient_source = core._canonicalize_examples(
            item["source"],
            assignment=assignment,
            transforms=transforms,
            heads=config.heads,
            head_dim=config.head_dim,
        )
        quotient_seen_inputs.append(quotient_source)
        quotient_seen_latent.append(item["latent"])
        seen_texts.extend(item["texts"])
        class_centroids = core._class_centroids(quotient_source, item["label"], config.classes)
        centroid_mean = class_centroids.mean(dim=0)
        quotient_seen_centroids.append(class_centroids - centroid_mean)
        quotient_seen_centroid_means.append(centroid_mean)

    quotient_seen_latent_tensor = torch.cat(quotient_seen_latent, dim=0)
    gpa_transforms, canonical_centroids = core._multiway_gpa(quotient_seen_centroids, iters=config.gpa_iters)
    aligned_seen_inputs = []
    for quotient_source, centroid_mean, gpa_transform in zip(quotient_seen_inputs, quotient_seen_centroid_means, gpa_transforms):
        aligned_seen_inputs.append((quotient_source - centroid_mean) @ gpa_transform)
    aligned_seen_inputs_tensor = torch.cat(aligned_seen_inputs, dim=0)

    learned_dictionary = core._spherical_kmeans(
        aligned_seen_inputs_tensor,
        k=config.atoms,
        iters=config.dictionary_iters,
    )
    atom_cost = 1.0 - (core._normalize_rows(learned_dictionary) @ core._normalize_rows(problem["true_atoms"]).T)
    atom_assignment = core._hungarian(atom_cost)
    seen_codes, _ = core._encode_with_dictionary(aligned_seen_inputs_tensor, learned_dictionary, topk=config.topk_atoms)
    seen_sidecar = sidecar._byte_sidecar_features(
        seen_texts,
        dim=config.sidecar_hash_dim,
        ngram_max=config.sidecar_ngram_max,
    )
    seen_alignment_sidecar, seen_alignment_profile = _sequence_aligned_sidecar_features(
        seen_texts,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        dim=config.alignment_hash_dim,
        token_scale=config.alignment_token_scale,
    )
    seen_joint = torch.cat(
        [
            seen_codes,
            float(config.sidecar_scale) * seen_sidecar,
            float(config.alignment_scale) * seen_alignment_sidecar,
            float(config.alignment_profile_scale) * seen_alignment_profile,
        ],
        dim=1,
    )
    decoder = core._fit_centered_ridge(seen_joint, quotient_seen_latent_tensor, config.ridge_lam)

    heldout_centroids = core._centroids_by_head(
        heldout_fewshot_norm["source"],
        heldout_fewshot_norm["label"],
        config.classes,
        config.heads,
        config.head_dim,
    )
    assignment_match, transforms_match = core._match_family_heads(heldout_centroids, anchor_centroids)
    heldout_match = core._canonicalize_examples(
        heldout_test_norm["source"],
        assignment=assignment_match,
        transforms=transforms_match,
        heads=config.heads,
        head_dim=config.head_dim,
    )
    heldout_match_centroids = core._class_centroids(heldout_match, heldout_test_norm["label"], config.classes)
    heldout_centroid_mean = heldout_match_centroids.mean(dim=0)
    heldout_gpa_transform = core.orthogonal_procrustes(heldout_match_centroids - heldout_centroid_mean, canonical_centroids)
    heldout_aligned = (heldout_match - heldout_centroid_mean) @ heldout_gpa_transform
    test_codes, test_selected_atoms = core._encode_with_dictionary(heldout_aligned, learned_dictionary, topk=config.topk_atoms)
    test_sidecar = sidecar._byte_sidecar_features(
        heldout_test_norm["texts"],
        dim=config.sidecar_hash_dim,
        ngram_max=config.sidecar_ngram_max,
    )
    test_alignment_sidecar, test_alignment_profile = _sequence_aligned_sidecar_features(
        heldout_test_norm["texts"],
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        dim=config.alignment_hash_dim,
        token_scale=config.alignment_token_scale,
    )
    test_joint = torch.cat(
        [
            test_codes,
            float(config.sidecar_scale) * test_sidecar,
            float(config.alignment_scale) * test_alignment_sidecar,
            float(config.alignment_profile_scale) * test_alignment_profile,
        ],
        dim=1,
    )
    predicted = core._predict_centered_ridge(decoder, test_joint)

    truth_assignment = core._true_assignment_to_anchor(problem, family_id=config.heldout_family, anchor_family=config.anchor_family)
    head_match_accuracy = core._head_match_accuracy(assignment_match, truth_assignment)
    atom_recovery = core._dictionary_atom_recovery(
        selected_atoms=test_selected_atoms,
        primary_atom=heldout_test_norm["primary_atom"],
        secondary_atom=heldout_test_norm["secondary_atom"],
        atom_assignment=atom_assignment,
    )
    dead_atom_rate = float((seen_codes.sum(dim=0) <= 1e-8).float().mean().item())
    codebook_perplexity = core._perplexity_from_codes(seen_codes)
    mean_sidecar_norm = float(test_sidecar.norm(dim=1).mean().item())
    mean_alignment_sidecar_norm = float(
        torch.cat(
            [
                float(config.alignment_scale) * test_alignment_sidecar,
                float(config.alignment_profile_scale) * test_alignment_profile,
            ],
            dim=1,
        ).norm(dim=1).mean().item()
    )
    return (
        predicted,
        head_match_accuracy,
        atom_recovery,
        dead_atom_rate,
        codebook_perplexity,
        mean_sidecar_norm,
        mean_alignment_sidecar_norm,
    )


def _fit_fewshot(
    config: ToyQuotientGPASparseDictionarySequenceAlignedSidecarConfig,
    *,
    heldout_fewshot_norm: dict[str, Any],
    heldout_test_norm: dict[str, Any],
) -> torch.Tensor:
    fewshot_model = core._fit_centered_ridge(heldout_fewshot_norm["source"], heldout_fewshot_norm["latent"], config.ridge_lam)
    return core._predict_centered_ridge(fewshot_model, heldout_test_norm["source"])


def _run_methods_for_shot(
    config: ToyQuotientGPASparseDictionarySequenceAlignedSidecarConfig,
    problem: dict[str, torch.Tensor],
    *,
    shot: int,
) -> list[dict[str, Any]]:
    base_cfg = _base_core_config(config)
    seen_family_ids = [family_id for family_id in range(config.families) if family_id != config.heldout_family]
    seen_items = [
        core._sample_family_examples(base_cfg, problem, family_id=family_id, shots_per_class=config.seen_shots_per_class, seed_offset=11_000)
        for family_id in seen_family_ids
    ]
    heldout_fewshot = core._sample_family_examples(base_cfg, problem, family_id=config.heldout_family, shots_per_class=shot, seed_offset=21_000)
    heldout_oracle = core._sample_family_examples(
        base_cfg,
        problem,
        family_id=config.heldout_family,
        shots_per_class=config.seen_shots_per_class,
        seed_offset=31_000,
    )
    heldout_test = core._sample_family_examples(
        base_cfg,
        problem,
        family_id=config.heldout_family,
        shots_per_class=config.test_examples_per_class,
        seed_offset=41_000,
    )

    source_tokenizer, target_tokenizer = iface.tokenizer_bridge._build_tokenizers()
    atom_words = iface._atom_to_word_map(config.atoms)
    calibration_texts = []
    for item in seen_items:
        calibration_texts.extend(iface._example_text(item, atom_words, seed=config.seed))
    remap_table = iface._build_remap_table_from_texts(
        calibration_texts,
        source=source_tokenizer,
        target=target_tokenizer,
        remap_capacity=config.remap_capacity,
    )

    seen_items = [
        sidecar._annotate_interface_with_texts(
            item,
            atom_words=atom_words,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            remap_table=remap_table,
            seed=config.seed,
        )
        for item in seen_items
    ]
    heldout_fewshot = sidecar._annotate_interface_with_texts(
        heldout_fewshot,
        atom_words=atom_words,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        remap_table=remap_table,
        seed=config.seed,
    )
    heldout_oracle = sidecar._annotate_interface_with_texts(
        heldout_oracle,
        atom_words=atom_words,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        remap_table=remap_table,
        seed=config.seed,
    )
    heldout_test = sidecar._annotate_interface_with_texts(
        heldout_test,
        atom_words=atom_words,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        remap_table=remap_table,
        seed=config.seed,
    )

    mix_matrix = iface._interface_mix_matrix(config.heads * config.head_dim, config.seed + 77)

    def build_variant(item: dict[str, Any], variant: str) -> dict[str, Any]:
        variant_source, interface_scale = iface._apply_interface_variant(
            item["source"],
            boundary_f1=item["boundary_f1"],
            remap_coverage=item["remap_coverage"],
            mix_matrix=mix_matrix,
            interface_strength=config.interface_strength,
            remap_recovery=config.remap_recovery,
            variant=variant,
        )
        return {**item, "source": variant_source, "interface_scale": interface_scale}

    variant_payloads: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for variant in ("token_id", "byte_span_remap", "oracle_interface"):
        variant_seen = [build_variant(item, variant) for item in seen_items]
        variant_fewshot = build_variant(heldout_fewshot, variant)
        variant_oracle = build_variant(heldout_oracle, variant)
        variant_test = build_variant(heldout_test, variant)
        variant_payloads[variant] = (variant_seen, variant_fewshot, variant_oracle, variant_test)

    rows = []
    baseline_accuracy = None
    baseline_mse = None

    for method in METHODS:
        if method.endswith("_token_id"):
            variant = "token_id"
        elif method.endswith("_byte_span_remap") or method.endswith("_remap"):
            variant = "byte_span_remap"
        else:
            variant = "oracle_interface"
        variant_seen, variant_fewshot, variant_oracle, variant_test = variant_payloads[variant]

        family_stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for item in variant_seen + [variant_oracle]:
            family_id = int(item["family"][0].item())
            family_stats[family_id] = core._family_center_scale(item["source"])

        seen_norm_items = []
        for item in variant_seen:
            family_id = int(item["family"][0].item())
            center, scale = family_stats[family_id]
            seen_norm_items.append({**item, "source": core._standardize(item["source"], center, scale)})
        heldout_center, heldout_scale = family_stats[config.heldout_family]
        heldout_fewshot_norm = {**variant_fewshot, "source": core._standardize(variant_fewshot["source"], heldout_center, heldout_scale)}
        heldout_test_norm = {**variant_test, "source": core._standardize(variant_test["source"], heldout_center, heldout_scale)}

        mean_alignment_sidecar_norm = None
        if method.startswith("heldout_fewshot_ridge"):
            predicted = _fit_fewshot(config, heldout_fewshot_norm=heldout_fewshot_norm, heldout_test_norm=heldout_test_norm)
            head_match_accuracy = None
            atom_recovery = None
            dead_atom_rate = None
            codebook_perplexity = None
            mean_sidecar_norm = None
            shared_basis = False
        elif "sequence_aligned_sidecar" in method:
            (
                predicted,
                head_match_accuracy,
                atom_recovery,
                dead_atom_rate,
                codebook_perplexity,
                mean_sidecar_norm,
                mean_alignment_sidecar_norm,
            ) = _compose_pipeline_with_sequence_aligned_sidecar(
                config,
                problem,
                seen_norm_items=seen_norm_items,
                heldout_fewshot_norm=heldout_fewshot_norm,
                heldout_test_norm=heldout_test_norm,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
            )
            shared_basis = True
        elif "byte_sidecar" in method:
            predicted, head_match_accuracy, atom_recovery, dead_atom_rate, codebook_perplexity, mean_sidecar_norm = sidecar._compose_pipeline_with_byte_sidecar(
                sidecar.ToyQuotientGPASparseDictionaryByteSidecarConfig(
                    seed=config.seed,
                    heads=config.heads,
                    head_dim=config.head_dim,
                    atoms=config.atoms,
                    classes=config.classes,
                    families=config.families,
                    heldout_family=config.heldout_family,
                    anchor_family=config.anchor_family,
                    seen_shots_per_class=config.seen_shots_per_class,
                    heldout_shots=config.heldout_shots,
                    test_examples_per_class=config.test_examples_per_class,
                    class_noise=config.class_noise,
                    source_noise=config.source_noise,
                    nuisance_rank=config.nuisance_rank,
                    nuisance_strength=config.nuisance_strength,
                    head_scale_jitter=config.head_scale_jitter,
                    head_bias_scale=config.head_bias_scale,
                    ridge_lam=config.ridge_lam,
                    gpa_iters=config.gpa_iters,
                    dictionary_iters=config.dictionary_iters,
                    topk_atoms=config.topk_atoms,
                    remap_capacity=config.remap_capacity,
                    interface_strength=config.interface_strength,
                    remap_recovery=config.remap_recovery,
                    sidecar_hash_dim=config.sidecar_hash_dim,
                    sidecar_ngram_max=config.sidecar_ngram_max,
                    sidecar_scale=config.sidecar_scale,
                ),
                problem,
                seen_norm_items=seen_norm_items,
                heldout_fewshot_norm=heldout_fewshot_norm,
                heldout_test_norm=heldout_test_norm,
            )
            shared_basis = True
        else:
            predicted, head_match_accuracy, atom_recovery, dead_atom_rate, codebook_perplexity, _ = iface._compose_pipeline(
                iface.ToyQuotientGPASparseDictionaryInterfaceStressConfig(
                    seed=config.seed,
                    heads=config.heads,
                    head_dim=config.head_dim,
                    atoms=config.atoms,
                    classes=config.classes,
                    families=config.families,
                    heldout_family=config.heldout_family,
                    anchor_family=config.anchor_family,
                    seen_shots_per_class=config.seen_shots_per_class,
                    heldout_shots=config.heldout_shots,
                    test_examples_per_class=config.test_examples_per_class,
                    class_noise=config.class_noise,
                    source_noise=config.source_noise,
                    nuisance_rank=config.nuisance_rank,
                    nuisance_strength=config.nuisance_strength,
                    head_scale_jitter=config.head_scale_jitter,
                    head_bias_scale=config.head_bias_scale,
                    ridge_lam=config.ridge_lam,
                    gpa_iters=config.gpa_iters,
                    dictionary_iters=config.dictionary_iters,
                    topk_atoms=config.topk_atoms,
                    remap_capacity=config.remap_capacity,
                    interface_strength=config.interface_strength,
                    remap_recovery=config.remap_recovery,
                ),
                problem,
                seen_norm_items=seen_norm_items,
                heldout_fewshot_norm=heldout_fewshot_norm,
                heldout_test_norm=heldout_test_norm,
            )
            mean_sidecar_norm = None
            shared_basis = True

        accuracy, mse, centroid_cosine = core._evaluate_prediction(
            predicted=predicted,
            target=heldout_test_norm["latent"],
            labels=heldout_test_norm["label"],
            problem=problem,
            config=base_cfg,
        )
        if method == "heldout_fewshot_ridge_token_id":
            baseline_accuracy = accuracy
            baseline_mse = mse

        rows.append(
            {
                "shot": int(shot),
                "method": method,
                "seed": int(config.seed),
                "heldout_family": int(config.heldout_family),
                "accuracy": accuracy,
                "mse": mse,
                "accuracy_delta_vs_token_id_fewshot": accuracy - float(baseline_accuracy),
                "mse_delta_vs_token_id_fewshot": mse - float(baseline_mse),
                "centroid_cosine": centroid_cosine,
                "mean_boundary_f1": float(variant_test["boundary_f1"].mean().item()),
                "mean_remap_coverage": float(variant_test["remap_coverage"].mean().item()),
                "mean_interface_noise_scale": float(variant_test["interface_scale"].mean().item()),
                "mean_sidecar_norm": mean_sidecar_norm,
                "mean_alignment_sidecar_norm": mean_alignment_sidecar_norm,
                "head_match_accuracy": head_match_accuracy,
                "atom_recovery": atom_recovery,
                "dead_atom_rate": dead_atom_rate,
                "codebook_perplexity": codebook_perplexity,
                "shared_basis": bool(shared_basis),
                "heldout_pairs_used": int(shot * config.classes),
            }
        )
    return rows


def run_experiment(config: ToyQuotientGPASparseDictionarySequenceAlignedSidecarConfig) -> dict[str, Any]:
    problem = core._build_problem(_base_core_config(config))
    rows = []
    for shot in config.heldout_shots:
        rows.extend(_run_methods_for_shot(config, problem, shot=int(shot)))
    config_payload = asdict(config)
    config_payload["heldout_shots"] = list(config.heldout_shots)
    summary: dict[str, Any] = {}
    for shot in config.heldout_shots:
        shot_rows = [row for row in rows if row["shot"] == int(shot)]
        best = min(shot_rows, key=lambda row: row["mse"])
        best_shared = min([row for row in shot_rows if row["shared_basis"]], key=lambda row: row["mse"])
        summary[str(int(shot))] = {
            "lowest_mse_method": best["method"],
            "lowest_mse": best["mse"],
            "best_shared_basis_method": best_shared["method"],
            "best_shared_basis_mse": best_shared["mse"],
        }
    return {
        "config": config_payload,
        "methods": list(METHODS),
        "rows": [{key: row[key] for key in ROW_KEY_ORDER} for row in rows],
        "summary": summary,
    }


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    config = payload["config"]
    rows = payload["rows"]
    lines = [
        "# Toy Quotient + GPA Sparse Dictionary Sequence-Aligned Sidecar",
        "",
        f"- seed: `{config['seed']}`",
        f"- held-out shot grid: `{config['heldout_shots']}`",
        f"- interface strength: `{config['interface_strength']}`",
        f"- remap recovery: `{config['remap_recovery']}`",
        f"- byte sidecar hash dim: `{config['sidecar_hash_dim']}`",
        f"- alignment hash dim: `{config['alignment_hash_dim']}`",
        f"- byte sidecar scale: `{config['sidecar_scale']}`",
        f"- alignment sidecar scale: `{config['alignment_scale']}`",
        f"- alignment profile scale: `{config['alignment_profile_scale']}`",
        "",
        "This toy keeps the current quotient+GPA+sparse-dictionary pipeline fixed and asks whether a sequence-aligned interface sidecar adds useful signal beyond the plain byte-sidecar branch under the same strong tokenizer-like corruption.",
        "",
        "References:",
        "- Cross-Tokenizer LLM Distillation through a Byte-Level Interface: https://arxiv.org/abs/2604.07466",
        "- DWA-KD: https://arxiv.org/abs/2602.21669",
        "- TokAlign: https://arxiv.org/abs/2506.03523",
        "- The Vision Wormhole: https://arxiv.org/abs/2602.15382",
        "",
        "| Shot | Method | Accuracy | MSE | dAcc vs token-id few-shot | dMSE vs token-id few-shot | Boundary F1 | Remap coverage | Interface noise | Byte sidecar norm | Alignment sidecar norm | Head-match acc | Atom recovery | Shared Basis |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "True" if value else "False"
        return f"{float(value):.4f}"

    for row in rows:
        lines.append(
            "| {shot} | {method} | {accuracy} | {mse} | {accuracy_delta_vs_token_id_fewshot} | {mse_delta_vs_token_id_fewshot} | {mean_boundary_f1} | {mean_remap_coverage} | {mean_interface_noise_scale} | {mean_sidecar_norm} | {mean_alignment_sidecar_norm} | {head_match_accuracy} | {atom_recovery} | {shared_basis} |".format(
                **{key: fmt(value) if key != "method" else value for key, value in row.items()}
            )
        )
    lines.extend(["", "## Best by Shot"])
    for shot, item in payload["summary"].items():
        lines.append(
            f"- {shot} shot/class: lowest MSE `{item['lowest_mse_method']}` ({item['lowest_mse']:.4f}); best shared-basis `{item['best_shared_basis_method']}` ({item['best_shared_basis_mse']:.4f})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out-family quotient+GPA sparse dictionary sequence-aligned sidecar toy.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=6)
    parser.add_argument("--atoms", type=int, default=10)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--families", type=int, default=5)
    parser.add_argument("--heldout-family", type=int, default=4)
    parser.add_argument("--anchor-family", type=int, default=0)
    parser.add_argument("--seen-shots-per-class", type=int, default=24)
    parser.add_argument("--heldout-shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--test-examples-per-class", type=int, default=64)
    parser.add_argument("--class-noise", type=float, default=0.16)
    parser.add_argument("--source-noise", type=float, default=0.03)
    parser.add_argument("--nuisance-rank", type=int, default=3)
    parser.add_argument("--nuisance-strength", type=float, default=0.22)
    parser.add_argument("--head-scale-jitter", type=float, default=0.18)
    parser.add_argument("--head-bias-scale", type=float, default=0.18)
    parser.add_argument("--ridge-lam", type=float, default=1e-2)
    parser.add_argument("--gpa-iters", type=int, default=8)
    parser.add_argument("--dictionary-iters", type=int, default=10)
    parser.add_argument("--topk-atoms", type=int, default=2)
    parser.add_argument("--remap-capacity", type=int, default=10)
    parser.add_argument("--interface-strength", type=float, default=2.5)
    parser.add_argument("--remap-recovery", type=float, default=0.70)
    parser.add_argument("--sidecar-hash-dim", type=int, default=16)
    parser.add_argument("--sidecar-ngram-max", type=int, default=2)
    parser.add_argument("--sidecar-scale", type=float, default=0.35)
    parser.add_argument("--alignment-hash-dim", type=int, default=16)
    parser.add_argument("--alignment-scale", type=float, default=0.30)
    parser.add_argument("--alignment-profile-scale", type=float, default=0.20)
    parser.add_argument("--alignment-token-scale", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyQuotientGPASparseDictionarySequenceAlignedSidecarConfig(
        seed=args.seed,
        heads=args.heads,
        head_dim=args.head_dim,
        atoms=args.atoms,
        classes=args.classes,
        families=args.families,
        heldout_family=args.heldout_family,
        anchor_family=args.anchor_family,
        seen_shots_per_class=args.seen_shots_per_class,
        heldout_shots=tuple(args.heldout_shots),
        test_examples_per_class=args.test_examples_per_class,
        class_noise=args.class_noise,
        source_noise=args.source_noise,
        nuisance_rank=args.nuisance_rank,
        nuisance_strength=args.nuisance_strength,
        head_scale_jitter=args.head_scale_jitter,
        head_bias_scale=args.head_bias_scale,
        ridge_lam=args.ridge_lam,
        gpa_iters=args.gpa_iters,
        dictionary_iters=args.dictionary_iters,
        topk_atoms=args.topk_atoms,
        remap_capacity=args.remap_capacity,
        interface_strength=args.interface_strength,
        remap_recovery=args.remap_recovery,
        sidecar_hash_dim=args.sidecar_hash_dim,
        sidecar_ngram_max=args.sidecar_ngram_max,
        sidecar_scale=args.sidecar_scale,
        alignment_hash_dim=args.alignment_hash_dim,
        alignment_scale=args.alignment_scale,
        alignment_profile_scale=args.alignment_profile_scale,
        alignment_token_scale=args.alignment_token_scale,
    )
    payload = run_experiment(config)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path = pathlib.Path(args.output_md) if args.output_md else output_path.with_suffix(".md")
    write_markdown_summary(payload, markdown_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
