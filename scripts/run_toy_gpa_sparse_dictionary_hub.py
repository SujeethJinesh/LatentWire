#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.procrustes import orthogonal_procrustes, ridge_projection


METHODS: tuple[str, ...] = (
    "heldout_fewshot_ridge",
    "global_seen_ridge",
    "multiway_gpa_canonical",
    "multiway_gpa_sparse_dictionary",
    "multiway_gpa_sparse_dictionary_repair",
    "oracle_family_ridge",
)


ROW_KEY_ORDER: tuple[str, ...] = (
    "shot",
    "method",
    "seed",
    "heldout_family",
    "accuracy",
    "mse",
    "accuracy_delta_vs_fewshot",
    "mse_delta_vs_fewshot",
    "centroid_cosine",
    "canonical_gap",
    "atom_recovery",
    "dead_atom_rate",
    "codebook_perplexity",
    "repair_accept_rate",
    "repair_help_rate",
    "repair_harm_rate",
    "shared_basis",
    "heldout_pairs_used",
)


@dataclass(frozen=True)
class ToyGPASparseDictionaryHubConfig:
    seed: int = 0
    dim: int = 20
    atoms: int = 10
    classes: int = 5
    families: int = 5
    heldout_family: int = 4
    seen_shots_per_class: int = 24
    heldout_shots: tuple[int, ...] = (1, 2, 4, 8)
    test_examples_per_class: int = 64
    class_noise: float = 0.18
    nuisance_rank: int = 4
    nuisance_strength: float = 0.32
    source_noise: float = 0.04
    scale_jitter: float = 0.08
    bias_scale: float = 0.30
    style_strength: float = 0.10
    ridge_lam: float = 1e-2
    gpa_iters: int = 8
    dictionary_iters: int = 10
    topk_atoms: int = 2
    repair_margin_threshold: float = 0.18
    repair_margin_gain: float = 0.01


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float32))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _fit_centered_ridge(source: torch.Tensor, target: torch.Tensor, ridge_lam: float) -> dict[str, torch.Tensor]:
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    weight = ridge_projection(source - source_mean, target - target_mean, lam=ridge_lam)
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "weight": weight,
    }


def _predict_centered_ridge(model: dict[str, torch.Tensor], source: torch.Tensor) -> torch.Tensor:
    return (source - model["source_mean"]) @ model["weight"] + model["target_mean"]


def _family_center_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-3)
    return center, scale


def _standardize(x: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x - center) / scale


def _class_centroids(x: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    default = x.mean(dim=0)
    centroids = []
    for class_id in range(num_classes):
        mask = labels.eq(class_id)
        centroids.append(x[mask].mean(dim=0) if bool(mask.any()) else default)
    return torch.stack(centroids, dim=0)


def _hungarian(cost: torch.Tensor) -> torch.Tensor:
    matrix = cost.detach().cpu().double().tolist()
    n = len(matrix)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = torch.empty(n, dtype=torch.long)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def _spherical_kmeans(x: torch.Tensor, *, k: int, iters: int) -> torch.Tensor:
    x_norm = _normalize_rows(x)
    init_idx = torch.linspace(0, x_norm.shape[0] - 1, steps=k).round().long()
    centers = x_norm[init_idx].clone()
    for _ in range(max(int(iters), 1)):
        scores = x_norm @ centers.T
        assignment = scores.argmax(dim=-1)
        max_scores = scores.max(dim=-1).values
        for atom_id in range(k):
            mask = assignment.eq(atom_id)
            if bool(mask.any()):
                centers[atom_id] = _normalize_rows(x_norm[mask].mean(dim=0, keepdim=True)).squeeze(0)
            else:
                centers[atom_id] = x_norm[max_scores.argmin()]
    return centers


def _encode_with_dictionary(x: torch.Tensor, dictionary: torch.Tensor, *, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_norm = _normalize_rows(x)
    dictionary_norm = _normalize_rows(dictionary)
    scores = (x_norm @ dictionary_norm.T).clamp_min(0.0)
    topk = max(1, min(int(topk), dictionary.shape[0]))
    top_values, top_indices = torch.topk(scores, k=topk, dim=-1)
    denom = top_values.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    weights = top_values / denom
    codes = torch.zeros_like(scores)
    codes.scatter_(1, top_indices, weights)
    fallback_mask = top_values.sum(dim=-1).le(1e-8)
    if bool(fallback_mask.any()):
        best = (x_norm @ dictionary_norm.T).argmax(dim=-1)
        codes[fallback_mask] = 0.0
        codes[fallback_mask, best[fallback_mask]] = 1.0
    return codes, top_indices


def _perplexity_from_codes(codes: torch.Tensor) -> float:
    usage = codes.sum(dim=0)
    probs = usage / usage.sum().clamp_min(1e-8)
    valid = probs > 0
    entropy = float((-(probs[valid] * torch.log(probs[valid]))).sum().item())
    return float(torch.exp(torch.tensor(entropy)).item())


def _code_entropy(codes: torch.Tensor) -> torch.Tensor:
    probs = codes.clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=-1)


def _build_problem(config: ToyGPASparseDictionaryHubConfig) -> dict[str, torch.Tensor]:
    if config.atoms < config.classes + 2:
        raise ValueError("atoms must be at least classes + 2")
    gen = _make_generator(config.seed)
    true_atoms = _normalize_rows(torch.randn(config.atoms, config.dim, generator=gen)) * 3.2
    nuisance_basis = _normalize_rows(torch.randn(config.nuisance_rank, config.dim, generator=gen))
    rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.families)], dim=0)
    scales = (1.0 + float(config.scale_jitter) * torch.randn(config.families, config.dim, generator=gen)).clamp(0.75, 1.25)
    bias = float(config.bias_scale) * torch.randn(config.families, config.dim, generator=gen)
    style = _normalize_rows(torch.randn(config.families, config.dim, generator=gen)) * float(config.style_strength)
    class_primary = torch.arange(config.classes, dtype=torch.long)
    secondary_pool = torch.arange(config.classes, config.atoms, dtype=torch.long)
    class_readout = true_atoms[class_primary]
    return {
        "true_atoms": true_atoms.float(),
        "nuisance_basis": nuisance_basis.float(),
        "rotations": rotations.float(),
        "scales": scales.float(),
        "bias": bias.float(),
        "style": style.float(),
        "class_primary": class_primary,
        "secondary_pool": secondary_pool,
        "class_readout": class_readout.float(),
    }


def _sample_family_examples(
    config: ToyGPASparseDictionaryHubConfig,
    problem: dict[str, torch.Tensor],
    *,
    family_id: int,
    shots_per_class: int,
    seed_offset: int,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset + family_id * 991)
    labels = torch.arange(config.classes, dtype=torch.long).repeat_interleave(shots_per_class)
    primary = problem["class_primary"][labels]
    secondary_index = torch.randint(problem["secondary_pool"].shape[0], (labels.shape[0],), generator=gen)
    secondary = problem["secondary_pool"][secondary_index]
    primary_weight = 1.05 + 0.25 * torch.rand(labels.shape[0], generator=gen)
    secondary_weight = 0.25 + 0.18 * torch.rand(labels.shape[0], generator=gen)
    latent = primary_weight.view(-1, 1) * problem["true_atoms"][primary]
    latent = latent + secondary_weight.view(-1, 1) * problem["true_atoms"][secondary]
    nuisance = float(config.nuisance_strength) * torch.randn(
        labels.shape[0],
        config.nuisance_rank,
        generator=gen,
        dtype=torch.float32,
    )
    latent = latent + nuisance @ problem["nuisance_basis"]
    latent = latent + float(config.class_noise) * torch.randn(latent.shape, generator=gen, dtype=torch.float32)
    latent = latent + 0.08 * torch.roll(latent, shifts=1, dims=-1)
    rotated = latent @ problem["rotations"][family_id]
    styled = rotated + problem["style"][family_id] * torch.tanh(latent)
    source = styled * problem["scales"][family_id] + problem["bias"][family_id]
    source = source + float(config.source_noise) * torch.randn(source.shape, generator=gen, dtype=torch.float32)
    family = torch.full((labels.shape[0],), family_id, dtype=torch.long)
    return {
        "source": source.float(),
        "latent": latent.float(),
        "label": labels.long(),
        "family": family,
        "primary_atom": primary.long(),
        "secondary_atom": secondary.long(),
    }


def _concat_examples(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.cat([item["source"] for item in items], dim=0),
        "latent": torch.cat([item["latent"] for item in items], dim=0),
        "label": torch.cat([item["label"] for item in items], dim=0),
        "family": torch.cat([item["family"] for item in items], dim=0),
        "primary_atom": torch.cat([item["primary_atom"] for item in items], dim=0),
        "secondary_atom": torch.cat([item["secondary_atom"] for item in items], dim=0),
    }


def _multiway_gpa(
    centroid_mats: list[torch.Tensor],
    *,
    iters: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    canonical = centroid_mats[0]
    transforms = [torch.eye(canonical.shape[1], dtype=canonical.dtype) for _ in centroid_mats]
    for _ in range(max(int(iters), 1)):
        aligned = []
        for idx, centroids in enumerate(centroid_mats):
            transform = orthogonal_procrustes(centroids, canonical)
            transforms[idx] = transform
            aligned.append(centroids @ transform)
        canonical = torch.stack(aligned, dim=0).mean(dim=0)
    return transforms, canonical


def _class_margin(predicted: torch.Tensor, class_readout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = predicted @ class_readout.T
    top2 = torch.topk(logits, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    pred_class = logits.argmax(dim=-1)
    return margin, pred_class


def _centroid_cosine(predicted: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, classes: int) -> float:
    pred_centroids = _class_centroids(predicted, labels, classes)
    tgt_centroids = _class_centroids(target, labels, classes)
    return float(F.cosine_similarity(pred_centroids, tgt_centroids, dim=-1).mean().item())


def _evaluate_prediction(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyGPASparseDictionaryHubConfig,
) -> tuple[float, float, float]:
    accuracy = float((predicted @ problem["class_readout"].T).argmax(dim=-1).eq(labels).float().mean().item())
    mse = float(F.mse_loss(predicted, target).item())
    centroid_cosine = _centroid_cosine(predicted, target, labels, config.classes)
    return accuracy, mse, centroid_cosine


def _dictionary_atom_recovery(
    *,
    selected_atoms: torch.Tensor,
    primary_atom: torch.Tensor,
    secondary_atom: torch.Tensor,
    atom_assignment: torch.Tensor,
) -> float:
    mapped = atom_assignment[selected_atoms]
    oracle = torch.stack([primary_atom, secondary_atom], dim=-1)
    recovery = []
    for idx in range(mapped.shape[0]):
        selected_set = set(int(item) for item in mapped[idx].tolist())
        oracle_set = set(int(item) for item in oracle[idx].tolist())
        recovery.append(len(selected_set & oracle_set) / float(len(oracle_set)))
    return float(sum(recovery) / max(len(recovery), 1))


def _run_methods_for_shot(
    config: ToyGPASparseDictionaryHubConfig,
    problem: dict[str, torch.Tensor],
    *,
    shot: int,
) -> list[dict[str, Any]]:
    seen_family_ids = [family_id for family_id in range(config.families) if family_id != config.heldout_family]
    seen_train_items = [
        _sample_family_examples(
            config,
            problem,
            family_id=family_id,
            shots_per_class=config.seen_shots_per_class,
            seed_offset=11_000,
        )
        for family_id in seen_family_ids
    ]
    seen_train = _concat_examples(seen_train_items)
    heldout_fewshot = _sample_family_examples(
        config,
        problem,
        family_id=config.heldout_family,
        shots_per_class=shot,
        seed_offset=21_000,
    )
    heldout_oracle = _sample_family_examples(
        config,
        problem,
        family_id=config.heldout_family,
        shots_per_class=config.seen_shots_per_class,
        seed_offset=31_000,
    )
    heldout_test = _sample_family_examples(
        config,
        problem,
        family_id=config.heldout_family,
        shots_per_class=config.test_examples_per_class,
        seed_offset=41_000,
    )

    family_stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for item in seen_train_items + [heldout_oracle]:
        family_id = int(item["family"][0].item())
        family_stats[family_id] = _family_center_scale(item["source"])

    seen_train_norm_items = []
    for item in seen_train_items:
        family_id = int(item["family"][0].item())
        center, scale = family_stats[family_id]
        seen_train_norm_items.append({**item, "source": _standardize(item["source"], center, scale)})
    seen_train_norm = _concat_examples(seen_train_norm_items)

    heldout_center, heldout_scale = family_stats[config.heldout_family]
    heldout_fewshot_norm = {**heldout_fewshot, "source": _standardize(heldout_fewshot["source"], heldout_center, heldout_scale)}
    heldout_oracle_norm = {**heldout_oracle, "source": _standardize(heldout_oracle["source"], heldout_center, heldout_scale)}
    heldout_test_norm = {**heldout_test, "source": _standardize(heldout_test["source"], heldout_center, heldout_scale)}

    fewshot_model = _fit_centered_ridge(heldout_fewshot_norm["source"], heldout_fewshot_norm["latent"], config.ridge_lam)
    global_model = _fit_centered_ridge(seen_train_norm["source"], seen_train_norm["latent"], config.ridge_lam)
    oracle_model = _fit_centered_ridge(heldout_oracle_norm["source"], heldout_oracle_norm["latent"], config.ridge_lam)

    seen_centroids = []
    seen_centroid_means = []
    for item in seen_train_norm_items:
        centroids = _class_centroids(item["source"], item["label"], config.classes)
        centroid_mean = centroids.mean(dim=0)
        seen_centroids.append(centroids - centroid_mean)
        seen_centroid_means.append(centroid_mean)
    gpa_transforms, canonical_centroids = _multiway_gpa(seen_centroids, iters=config.gpa_iters)

    heldout_centroids = _class_centroids(heldout_fewshot_norm["source"], heldout_fewshot_norm["label"], config.classes)
    heldout_centroid_mean = heldout_centroids.mean(dim=0)
    heldout_transform = orthogonal_procrustes(heldout_centroids - heldout_centroid_mean, canonical_centroids)

    aligned_seen_inputs = []
    aligned_seen_latent = []
    for item, centroid_mean, transform in zip(seen_train_norm_items, seen_centroid_means, gpa_transforms):
        aligned_seen_inputs.append((item["source"] - centroid_mean) @ transform)
        aligned_seen_latent.append(item["latent"])
    aligned_seen_inputs_tensor = torch.cat(aligned_seen_inputs, dim=0)
    aligned_seen_latent_tensor = torch.cat(aligned_seen_latent, dim=0)
    aligned_heldout_test = (heldout_test_norm["source"] - heldout_centroid_mean) @ heldout_transform

    canonical_model = _fit_centered_ridge(aligned_seen_inputs_tensor, aligned_seen_latent_tensor, config.ridge_lam)

    learned_dictionary = _spherical_kmeans(
        aligned_seen_inputs_tensor,
        k=config.atoms,
        iters=config.dictionary_iters,
    )
    atom_cost = 1.0 - (_normalize_rows(learned_dictionary) @ _normalize_rows(problem["true_atoms"]).T)
    atom_assignment = _hungarian(atom_cost)
    seen_codes, _ = _encode_with_dictionary(aligned_seen_inputs_tensor, learned_dictionary, topk=config.topk_atoms)
    test_codes, test_selected_atoms = _encode_with_dictionary(aligned_heldout_test, learned_dictionary, topk=config.topk_atoms)
    sparse_decoder = _fit_centered_ridge(seen_codes, aligned_seen_latent_tensor, config.ridge_lam)
    sparse_pred = _predict_centered_ridge(sparse_decoder, test_codes)

    seen_sparse_pred = _predict_centered_ridge(sparse_decoder, seen_codes)
    repair_features_seen = torch.cat([seen_sparse_pred, seen_codes], dim=-1)
    repair_delta_model = _fit_centered_ridge(
        repair_features_seen,
        aligned_seen_latent_tensor - seen_sparse_pred,
        config.ridge_lam,
    )
    repair_features_test = torch.cat([sparse_pred, test_codes], dim=-1)
    repaired_pred = sparse_pred + _predict_centered_ridge(repair_delta_model, repair_features_test)

    seen_repaired_pred = seen_sparse_pred + _predict_centered_ridge(repair_delta_model, repair_features_seen)
    seen_base_margin, _ = _class_margin(seen_sparse_pred, problem["class_readout"])
    seen_repaired_margin, _ = _class_margin(seen_repaired_pred, problem["class_readout"])
    seen_delta_norm = (seen_repaired_pred - seen_sparse_pred).norm(dim=-1)
    seen_gate_features = torch.stack(
        [
            seen_base_margin,
            seen_repaired_margin - seen_base_margin,
            _code_entropy(seen_codes),
            seen_delta_norm,
        ],
        dim=-1,
    )
    seen_help_label = (
        (seen_repaired_pred - aligned_seen_latent_tensor).pow(2).mean(dim=-1)
        < (seen_sparse_pred - aligned_seen_latent_tensor).pow(2).mean(dim=-1) - 1e-8
    ).float()
    repair_verifier = _fit_centered_ridge(seen_gate_features, seen_help_label.view(-1, 1), config.ridge_lam)

    base_margin, _ = _class_margin(sparse_pred, problem["class_readout"])
    repaired_margin, _ = _class_margin(repaired_pred, problem["class_readout"])
    delta_norm = (repaired_pred - sparse_pred).norm(dim=-1)
    gate_features = torch.stack(
        [
            base_margin,
            repaired_margin - base_margin,
            _code_entropy(test_codes),
            delta_norm,
        ],
        dim=-1,
    )
    repair_score = _predict_centered_ridge(repair_verifier, gate_features).squeeze(-1)
    repair_gate = repair_score > 0.5
    repair_gate = repair_gate & (base_margin <= float(config.repair_margin_threshold))
    repair_gate = repair_gate & ((repaired_margin - base_margin) >= float(config.repair_margin_gain))
    repaired_final = torch.where(repair_gate.view(-1, 1), repaired_pred, sparse_pred)

    predictions: dict[str, torch.Tensor] = {
        "heldout_fewshot_ridge": _predict_centered_ridge(fewshot_model, heldout_test_norm["source"]),
        "global_seen_ridge": _predict_centered_ridge(global_model, heldout_test_norm["source"]),
        "multiway_gpa_canonical": _predict_centered_ridge(canonical_model, aligned_heldout_test),
        "multiway_gpa_sparse_dictionary": sparse_pred,
        "multiway_gpa_sparse_dictionary_repair": repaired_final,
        "oracle_family_ridge": _predict_centered_ridge(oracle_model, heldout_test_norm["source"]),
    }

    aligned_centroids = {
        "multiway_gpa_canonical": _class_centroids(aligned_heldout_test, heldout_test_norm["label"], config.classes),
        "multiway_gpa_sparse_dictionary": _class_centroids(test_codes @ learned_dictionary, heldout_test_norm["label"], config.classes),
        "multiway_gpa_sparse_dictionary_repair": _class_centroids(test_codes @ learned_dictionary, heldout_test_norm["label"], config.classes),
    }

    baseline_accuracy, baseline_mse, _ = _evaluate_prediction(
        predicted=predictions["heldout_fewshot_ridge"],
        target=heldout_test_norm["latent"],
        labels=heldout_test_norm["label"],
        problem=problem,
        config=config,
    )

    sparse_accuracy = (sparse_pred @ problem["class_readout"].T).argmax(dim=-1).eq(heldout_test_norm["label"])
    repaired_accuracy = (repaired_final @ problem["class_readout"].T).argmax(dim=-1).eq(heldout_test_norm["label"])
    repair_help = repair_gate & repaired_accuracy & ~sparse_accuracy
    repair_harm = repair_gate & sparse_accuracy & ~repaired_accuracy
    atom_recovery = _dictionary_atom_recovery(
        selected_atoms=test_selected_atoms,
        primary_atom=heldout_test_norm["primary_atom"],
        secondary_atom=heldout_test_norm["secondary_atom"],
        atom_assignment=atom_assignment,
    )
    dead_atom_rate = float((seen_codes.sum(dim=0) <= 1e-8).float().mean().item())
    codebook_perplexity = _perplexity_from_codes(seen_codes)

    rows = []
    for method in METHODS:
        predicted = predictions[method]
        accuracy, mse, centroid_cosine = _evaluate_prediction(
            predicted=predicted,
            target=heldout_test_norm["latent"],
            labels=heldout_test_norm["label"],
            problem=problem,
            config=config,
        )
        if method in aligned_centroids:
            canonical_gap = float(F.mse_loss(aligned_centroids[method], canonical_centroids).item())
            shared_basis = True
        else:
            canonical_gap = float(F.mse_loss(_class_centroids(predicted, heldout_test_norm["label"], config.classes), problem["class_readout"]).item())
            shared_basis = False
        row = {
            "shot": int(shot),
            "method": method,
            "seed": int(config.seed),
            "heldout_family": int(config.heldout_family),
            "accuracy": accuracy,
            "mse": mse,
            "accuracy_delta_vs_fewshot": accuracy - baseline_accuracy,
            "mse_delta_vs_fewshot": mse - baseline_mse,
            "centroid_cosine": centroid_cosine,
            "canonical_gap": canonical_gap,
            "atom_recovery": atom_recovery if "sparse_dictionary" in method else None,
            "dead_atom_rate": dead_atom_rate if "sparse_dictionary" in method else None,
            "codebook_perplexity": codebook_perplexity if "sparse_dictionary" in method else None,
            "repair_accept_rate": float(repair_gate.float().mean().item()) if method == "multiway_gpa_sparse_dictionary_repair" else 0.0,
            "repair_help_rate": float(repair_help.float().mean().item()) if method == "multiway_gpa_sparse_dictionary_repair" else 0.0,
            "repair_harm_rate": float(repair_harm.float().mean().item()) if method == "multiway_gpa_sparse_dictionary_repair" else 0.0,
            "shared_basis": bool(shared_basis),
            "heldout_pairs_used": int(shot * config.classes if method != "oracle_family_ridge" else config.seen_shots_per_class * config.classes),
        }
        rows.append(row)
    return rows


def run_experiment(config: ToyGPASparseDictionaryHubConfig) -> dict[str, Any]:
    problem = _build_problem(config)
    rows = []
    for shot in config.heldout_shots:
        rows.extend(_run_methods_for_shot(config, problem, shot=int(shot)))

    config_payload = asdict(config)
    config_payload["heldout_shots"] = list(config.heldout_shots)

    summary: dict[str, Any] = {}
    for shot in config.heldout_shots:
        shot_rows = [row for row in rows if row["shot"] == int(shot)]
        non_oracle_rows = [row for row in shot_rows if row["method"] != "oracle_family_ridge"]
        best_accuracy = max(non_oracle_rows, key=lambda row: (row["accuracy"], -row["mse"]))
        lowest_mse = min(non_oracle_rows, key=lambda row: row["mse"])
        best_shared = min([row for row in non_oracle_rows if row["shared_basis"]], key=lambda row: row["mse"])
        summary[str(int(shot))] = {
            "best_non_oracle_accuracy_method": best_accuracy["method"],
            "best_non_oracle_accuracy": best_accuracy["accuracy"],
            "best_non_oracle_accuracy_mse": best_accuracy["mse"],
            "lowest_mse_non_oracle_method": lowest_mse["method"],
            "lowest_mse_non_oracle": lowest_mse["mse"],
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
    summary = payload["summary"]
    lines = [
        "# Toy GPA Sparse Dictionary Hub Sweep",
        "",
        f"- seed: `{config['seed']}`",
        f"- dim: `{config['dim']}`",
        f"- atoms: `{config['atoms']}`",
        f"- classes: `{config['classes']}`",
        f"- families: `{config['families']}`",
        f"- held-out family: `{config['heldout_family']}`",
        f"- seen shots / class: `{config['seen_shots_per_class']}`",
        f"- held-out shot grid: `{config['heldout_shots']}`",
        "",
        "This toy tests the next positive-method branch directly: GPA-style canonicalization first, then a sparse shared dictionary, then one verifier-gated repair step with routing frozen by construction.",
        "",
        "References:",
        "- Multi-Way Representation Alignment: https://arxiv.org/abs/2602.06205",
        "- Universal Sparse Autoencoders: https://arxiv.org/abs/2502.03714",
        "- Delta-Crosscoder: https://arxiv.org/abs/2603.04426",
        "",
        "| Shot | Method | Accuracy | MSE | dAcc vs few-shot | dMSE vs few-shot | Atom rec | Dead atoms | Perplexity | Repair accept | Repair help | Repair harm | Shared Basis |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "True" if value else "False"
        return f"{float(value):.4f}"
    for row in rows:
        lines.append(
            "| {shot} | {method} | {accuracy} | {mse} | {accuracy_delta_vs_fewshot} | {mse_delta_vs_fewshot} | {atom_recovery} | {dead_atom_rate} | {codebook_perplexity} | {repair_accept_rate} | {repair_help_rate} | {repair_harm_rate} | {shared_basis} |".format(
                **{key: fmt(value) if key != "method" else value for key, value in row.items()}
            )
        )
    lines.extend(["", "## Best Non-Oracle by Shot"])
    for shot, item in summary.items():
        lines.append(
            f"- {shot} shot/class: best accuracy `{item['best_non_oracle_accuracy_method']}` ({item['best_non_oracle_accuracy']:.4f} acc, {item['best_non_oracle_accuracy_mse']:.4f} MSE); "
            f"lowest non-oracle MSE `{item['lowest_mse_non_oracle_method']}` ({item['lowest_mse_non_oracle']:.4f}); "
            f"best shared-basis `{item['best_shared_basis_method']}` ({item['best_shared_basis_mse']:.4f} MSE)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out-family GPA sparse dictionary hub toy.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--output-md", help="Markdown output path. Defaults to the JSON path with a .md suffix.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--atoms", type=int, default=10)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--families", type=int, default=5)
    parser.add_argument("--heldout-family", type=int, default=4)
    parser.add_argument("--seen-shots-per-class", type=int, default=24)
    parser.add_argument("--heldout-shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--test-examples-per-class", type=int, default=64)
    parser.add_argument("--class-noise", type=float, default=0.18)
    parser.add_argument("--nuisance-rank", type=int, default=4)
    parser.add_argument("--nuisance-strength", type=float, default=0.32)
    parser.add_argument("--source-noise", type=float, default=0.04)
    parser.add_argument("--scale-jitter", type=float, default=0.08)
    parser.add_argument("--bias-scale", type=float, default=0.30)
    parser.add_argument("--style-strength", type=float, default=0.10)
    parser.add_argument("--ridge-lam", type=float, default=1e-2)
    parser.add_argument("--gpa-iters", type=int, default=8)
    parser.add_argument("--dictionary-iters", type=int, default=10)
    parser.add_argument("--topk-atoms", type=int, default=2)
    parser.add_argument("--repair-margin-threshold", type=float, default=0.18)
    parser.add_argument("--repair-margin-gain", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyGPASparseDictionaryHubConfig(
        seed=args.seed,
        dim=args.dim,
        atoms=args.atoms,
        classes=args.classes,
        families=args.families,
        heldout_family=args.heldout_family,
        seen_shots_per_class=args.seen_shots_per_class,
        heldout_shots=tuple(args.heldout_shots),
        test_examples_per_class=args.test_examples_per_class,
        class_noise=args.class_noise,
        nuisance_rank=args.nuisance_rank,
        nuisance_strength=args.nuisance_strength,
        source_noise=args.source_noise,
        scale_jitter=args.scale_jitter,
        bias_scale=args.bias_scale,
        style_strength=args.style_strength,
        ridge_lam=args.ridge_lam,
        gpa_iters=args.gpa_iters,
        dictionary_iters=args.dictionary_iters,
        topk_atoms=args.topk_atoms,
        repair_margin_threshold=args.repair_margin_threshold,
        repair_margin_gain=args.repair_margin_gain,
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
