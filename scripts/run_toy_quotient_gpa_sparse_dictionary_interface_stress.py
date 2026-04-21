#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_toy_quotient_gpa_sparse_dictionary as core
from scripts import run_toy_tokenizer_frontier_bridge as tokenizer_bridge


METHODS: tuple[str, ...] = (
    "heldout_fewshot_ridge_token_id",
    "heldout_fewshot_ridge_byte_span_remap",
    "quotient_gpa_sparse_dictionary_token_id",
    "quotient_gpa_sparse_dictionary_byte_span_remap",
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
    "head_match_accuracy",
    "atom_recovery",
    "dead_atom_rate",
    "codebook_perplexity",
    "shared_basis",
    "heldout_pairs_used",
)


@dataclass(frozen=True)
class ToyQuotientGPASparseDictionaryInterfaceStressConfig:
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


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _interface_mix_matrix(dim: int, seed: int) -> torch.Tensor:
    gen = _make_generator(seed)
    mix = torch.randn(dim, dim, generator=gen, dtype=torch.float32) / math.sqrt(dim)
    mix = mix - torch.diag(torch.diag(mix))
    return mix


def _atom_to_word_map(atoms: int) -> list[str]:
    source_words = list(tokenizer_bridge._SOURCE_WORDS)
    if atoms > len(source_words):
        raise ValueError("Not enough source words to assign atoms")
    return source_words[:atoms]


def _example_text(item: dict[str, torch.Tensor], atom_words: list[str], *, seed: int) -> list[str]:
    labels = item["label"].tolist()
    primary = item["primary_atom"].tolist()
    secondary = item["secondary_atom"].tolist()
    texts = []
    for idx, (label, primary_atom, secondary_atom) in enumerate(zip(labels, primary, secondary)):
        joiner_a = tokenizer_bridge._JOINERS[(seed + idx + label) % len(tokenizer_bridge._JOINERS)]
        joiner_b = tokenizer_bridge._JOINERS[(seed + idx + secondary_atom + 1) % len(tokenizer_bridge._JOINERS)]
        number = tokenizer_bridge._NUMBERS[(seed * 3 + idx + label) % len(tokenizer_bridge._NUMBERS)]
        texts.append(f"{atom_words[primary_atom]}{joiner_a}{atom_words[secondary_atom]}{joiner_b}{number}")
    return texts


def _build_remap_table_from_texts(
    texts: list[str],
    *,
    source: tokenizer_bridge.ToyTokenizer,
    target: tokenizer_bridge.ToyTokenizer,
    remap_capacity: int,
) -> dict[str, tuple[str, ...]]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(source.segment(text))

    scored: list[tuple[float, str, tuple[str, ...]]] = []
    for token, count in counts.items():
        target_tokens = tuple(target.segment(token))
        if len(target_tokens) <= 1:
            continue
        savings = float(len(target_tokens) - 1)
        score = float(count) * savings - 0.25 * len(token)
        if score > 0.0:
            scored.append((score, token, target_tokens))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return {token: target_tokens for _, token, target_tokens in scored[: max(0, int(remap_capacity))]}


def _annotate_interface(
    item: dict[str, torch.Tensor],
    *,
    atom_words: list[str],
    source_tokenizer: tokenizer_bridge.ToyTokenizer,
    target_tokenizer: tokenizer_bridge.ToyTokenizer,
    remap_table: dict[str, tuple[str, ...]],
    seed: int,
) -> dict[str, Any]:
    texts = _example_text(item, atom_words, seed=seed)
    boundary_f1 = []
    remap_coverage = []
    for text in texts:
        source_tokens = source_tokenizer.segment(text)
        target_tokens = target_tokenizer.segment(text)
        f1 = tokenizer_bridge._boundary_f1(
            tokenizer_bridge._boundary_positions(text, source_tokens),
            tokenizer_bridge._boundary_positions(text, target_tokens),
        )
        coverage = sum(1 for token in source_tokens if token in remap_table) / max(len(source_tokens), 1)
        boundary_f1.append(float(f1))
        remap_coverage.append(float(coverage))
    return {
        **item,
        "boundary_f1": torch.tensor(boundary_f1, dtype=torch.float32),
        "remap_coverage": torch.tensor(remap_coverage, dtype=torch.float32),
    }


def _apply_interface_variant(
    source_values: torch.Tensor,
    *,
    boundary_f1: torch.Tensor,
    remap_coverage: torch.Tensor,
    mix_matrix: torch.Tensor,
    interface_strength: float,
    remap_recovery: float,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_scale = float(interface_strength) * (1.0 - boundary_f1)
    if variant == "token_id":
        scale = base_scale
    elif variant == "byte_span_remap":
        scale = base_scale * (1.0 - float(remap_recovery) * remap_coverage)
    elif variant == "oracle_interface":
        scale = torch.zeros_like(base_scale)
    else:
        raise ValueError(f"Unknown interface variant: {variant}")
    mixed = source_values @ mix_matrix
    corrupted = source_values + scale.unsqueeze(-1) * mixed
    return corrupted, scale


def _compose_pipeline(
    config: ToyQuotientGPASparseDictionaryInterfaceStressConfig,
    problem: dict[str, torch.Tensor],
    *,
    seen_norm_items: list[dict[str, Any]],
    heldout_fewshot_norm: dict[str, Any],
    heldout_test_norm: dict[str, Any],
) -> tuple[torch.Tensor, float, float, float, float, float]:
    anchor_item = next(item for item in seen_norm_items if int(item["family"][0].item()) == config.anchor_family)
    anchor_centroids = core._centroids_by_head(anchor_item["source"], anchor_item["label"], config.classes, config.heads, config.head_dim)

    quotient_seen_inputs = []
    quotient_seen_latent = []
    quotient_seen_centroids = []
    quotient_seen_centroid_means = []
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
        class_centroids = core._class_centroids(quotient_source, item["label"], config.classes)
        centroid_mean = class_centroids.mean(dim=0)
        quotient_seen_centroids.append(class_centroids - centroid_mean)
        quotient_seen_centroid_means.append(centroid_mean)

    quotient_seen_inputs_tensor = torch.cat(quotient_seen_inputs, dim=0)
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
    sparse_decoder = core._fit_centered_ridge(seen_codes, quotient_seen_latent_tensor, config.ridge_lam)

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
    sparse_pred = core._predict_centered_ridge(sparse_decoder, test_codes)

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
    return sparse_pred, head_match_accuracy, atom_recovery, dead_atom_rate, codebook_perplexity, float(F.mse_loss(core._class_centroids(heldout_aligned, heldout_test_norm["label"], config.classes), canonical_centroids).item())


def _fit_fewshot(
    config: ToyQuotientGPASparseDictionaryInterfaceStressConfig,
    *,
    heldout_fewshot_norm: dict[str, Any],
    heldout_test_norm: dict[str, Any],
) -> torch.Tensor:
    fewshot_model = core._fit_centered_ridge(heldout_fewshot_norm["source"], heldout_fewshot_norm["latent"], config.ridge_lam)
    return core._predict_centered_ridge(fewshot_model, heldout_test_norm["source"])


def _run_methods_for_shot(
    config: ToyQuotientGPASparseDictionaryInterfaceStressConfig,
    problem: dict[str, torch.Tensor],
    *,
    shot: int,
) -> list[dict[str, Any]]:
    seen_family_ids = [family_id for family_id in range(config.families) if family_id != config.heldout_family]
    seen_items = [
        core._sample_family_examples(
            core.ToyQuotientGPASparseDictionaryConfig(
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
            ),
            problem,
            family_id=family_id,
            shots_per_class=config.seen_shots_per_class,
            seed_offset=11_000,
        )
        for family_id in seen_family_ids
    ]
    heldout_fewshot = core._sample_family_examples(
        core.ToyQuotientGPASparseDictionaryConfig(
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
        ),
        problem,
        family_id=config.heldout_family,
        shots_per_class=shot,
        seed_offset=21_000,
    )
    heldout_oracle = core._sample_family_examples(
        core.ToyQuotientGPASparseDictionaryConfig(
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
        ),
        problem,
        family_id=config.heldout_family,
        shots_per_class=config.seen_shots_per_class,
        seed_offset=31_000,
    )
    heldout_test = core._sample_family_examples(
        core.ToyQuotientGPASparseDictionaryConfig(
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
        ),
        problem,
        family_id=config.heldout_family,
        shots_per_class=config.test_examples_per_class,
        seed_offset=41_000,
    )

    source_tokenizer, target_tokenizer = tokenizer_bridge._build_tokenizers()
    atom_words = _atom_to_word_map(config.atoms)
    calibration_texts = []
    for item in seen_items:
        calibration_texts.extend(_example_text(item, atom_words, seed=config.seed))
    remap_table = _build_remap_table_from_texts(
        calibration_texts,
        source=source_tokenizer,
        target=target_tokenizer,
        remap_capacity=config.remap_capacity,
    )

    seen_items = [
        _annotate_interface(
            item,
            atom_words=atom_words,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            remap_table=remap_table,
            seed=config.seed,
        )
        for item in seen_items
    ]
    heldout_fewshot = _annotate_interface(
        heldout_fewshot,
        atom_words=atom_words,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        remap_table=remap_table,
        seed=config.seed,
    )
    heldout_oracle = _annotate_interface(
        heldout_oracle,
        atom_words=atom_words,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        remap_table=remap_table,
        seed=config.seed,
    )
    heldout_test = _annotate_interface(
        heldout_test,
        atom_words=atom_words,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        remap_table=remap_table,
        seed=config.seed,
    )

    mix_matrix = _interface_mix_matrix(config.heads * config.head_dim, config.seed + 77)

    def build_variant(item: dict[str, Any], variant: str) -> dict[str, Any]:
        variant_source, interface_scale = _apply_interface_variant(
            item["source"],
            boundary_f1=item["boundary_f1"],
            remap_coverage=item["remap_coverage"],
            mix_matrix=mix_matrix,
            interface_strength=config.interface_strength,
            remap_recovery=config.remap_recovery,
            variant=variant,
        )
        return {
            **item,
            "source": variant_source,
            "interface_scale": interface_scale,
        }

    variant_payloads: dict[str, tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for variant in ("token_id", "byte_span_remap", "oracle_interface"):
        variant_seen = [build_variant(item, variant) for item in seen_items]
        variant_fewshot = build_variant(heldout_fewshot, variant)
        variant_oracle = build_variant(heldout_oracle, variant)
        variant_test = build_variant(heldout_test, variant)
        variant_payloads[variant] = (variant_seen, variant_fewshot, variant_oracle, variant_test)

    rows = []
    baseline_pred = None
    baseline_accuracy = None
    baseline_mse = None

    for method in METHODS:
        if method.endswith("_token_id"):
            variant = "token_id"
        elif method.endswith("_byte_span_remap"):
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

        if method.startswith("heldout_fewshot_ridge"):
            predicted = _fit_fewshot(config, heldout_fewshot_norm=heldout_fewshot_norm, heldout_test_norm=heldout_test_norm)
            head_match_accuracy = None
            atom_recovery = None
            dead_atom_rate = None
            codebook_perplexity = None
            shared_basis = False
        else:
            predicted, head_match_accuracy, atom_recovery, dead_atom_rate, codebook_perplexity, _ = _compose_pipeline(
                config,
                problem,
                seen_norm_items=seen_norm_items,
                heldout_fewshot_norm=heldout_fewshot_norm,
                heldout_test_norm=heldout_test_norm,
            )
            shared_basis = True

        accuracy, mse, centroid_cosine = core._evaluate_prediction(
            predicted=predicted,
            target=heldout_test_norm["latent"],
            labels=heldout_test_norm["label"],
            problem=problem,
            config=core.ToyQuotientGPASparseDictionaryConfig(
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
            ),
        )
        if method == "heldout_fewshot_ridge_token_id":
            baseline_pred = predicted
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
                "head_match_accuracy": head_match_accuracy,
                "atom_recovery": atom_recovery,
                "dead_atom_rate": dead_atom_rate,
                "codebook_perplexity": codebook_perplexity,
                "shared_basis": bool(shared_basis),
                "heldout_pairs_used": int(shot * config.classes),
            }
        )
    return rows


def run_experiment(config: ToyQuotientGPASparseDictionaryInterfaceStressConfig) -> dict[str, Any]:
    base_cfg = core.ToyQuotientGPASparseDictionaryConfig(
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
    problem = core._build_problem(base_cfg)
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
        "# Toy Quotient + GPA Sparse Dictionary Interface Stress",
        "",
        f"- seed: `{config['seed']}`",
        f"- held-out shot grid: `{config['heldout_shots']}`",
        f"- remap capacity: `{config['remap_capacity']}`",
        f"- interface strength: `{config['interface_strength']}`",
        f"- remap recovery: `{config['remap_recovery']}`",
        "",
        "This toy stresses the current best low-shot shared-basis lane under strong tokenizer-like interface corruption. It compares raw token-id interface transfer against a learned shared byte/span remap and an oracle interface while keeping the quotient+GPA+sparse-dictionary pipeline fixed.",
        "",
        "References:",
        "- TokAlign: https://arxiv.org/abs/2506.03523",
        "- Byte Latent Transformer: https://arxiv.org/abs/2412.09871",
        "- Complete Characterization of Gauge Symmetries in Transformer Architectures: https://openreview.net/forum?id=KrkbYbK0cH",
        "",
        "| Shot | Method | Accuracy | MSE | dAcc vs token-id few-shot | dMSE vs token-id few-shot | Boundary F1 | Remap coverage | Interface noise | Head-match acc | Atom recovery | Shared Basis |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "True" if value else "False"
        return f"{float(value):.4f}"

    for row in rows:
        lines.append(
            "| {shot} | {method} | {accuracy} | {mse} | {accuracy_delta_vs_token_id_fewshot} | {mse_delta_vs_token_id_fewshot} | {mean_boundary_f1} | {mean_remap_coverage} | {mean_interface_noise_scale} | {head_match_accuracy} | {atom_recovery} | {shared_basis} |".format(
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
    parser = argparse.ArgumentParser(description="Held-out-family quotient+GPA sparse dictionary interface stress toy.")
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
    parser.add_argument("--interface-strength", type=float, default=0.85)
    parser.add_argument("--remap-recovery", type=float, default=0.70)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyQuotientGPASparseDictionaryInterfaceStressConfig(
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
