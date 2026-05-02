#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


METHODS: tuple[str, ...] = (
    "raw_ridge",
    "shared_feature_only",
    "route_atom_only",
    "stacked_feature_atom",
    "protected_stacked_feature_atom",
    "oracle",
)


@dataclass(frozen=True)
class ToyFeatureAtomStackBridgeConfig:
    seed: int = 0
    train_examples: int = 144
    test_examples: int = 96
    dim: int = 16
    shared_features: int = 6
    route_families: int = 4
    atoms_per_family: int = 4
    source_private_features: int = 4
    target_private_features: int = 4
    shared_sparsity: float = 0.35
    private_sparsity: float = 0.20
    shared_scale: float = 1.15
    atom_scale: float = 1.8
    private_scale: float = 0.45
    noise: float = 0.04
    shared_iters: int = 8
    atom_iters: int = 8
    bridge_lam: float = 1e-2
    dictionary_lam: float = 1e-3
    codebook_temp: float = 0.35
    protected_shared: int = 2
    protected_atoms: int = 3
    classes: int = 5


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float32))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _normalize_rows(matrix: torch.Tensor) -> torch.Tensor:
    norms = matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return matrix / norms


def _soft_threshold_rows(x: torch.Tensor, active: int) -> torch.Tensor:
    if active >= x.shape[1]:
        return x
    values, indices = torch.topk(x.abs(), k=active, dim=1)
    del values
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return torch.where(mask, x, torch.zeros_like(x))


def _bytes_for_values(count: int, bits: float) -> float:
    return float(math.ceil(count * bits / 8.0))


def _bytes_for_matrix(rows: int, cols: int) -> float:
    return float(rows * cols * 4)


def _entropy_from_counts(counts: torch.Tensor) -> float:
    total = counts.sum().clamp_min(1e-8)
    probs = counts / total
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
    return float(entropy.item())


def _perplexity_from_counts(counts: torch.Tensor) -> float:
    return float(math.exp(_entropy_from_counts(counts)))


def _match_recovery(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    if reference.numel() == 0 or candidate.numel() == 0:
        return 0.0
    ref = F.normalize(reference, dim=1)
    cand = F.normalize(candidate, dim=1)
    m = min(ref.shape[0], cand.shape[0])
    cost = 1.0 - (ref[:m] @ cand[:m].T)
    assignment = _hungarian(cost)
    aligned = cand[:m][assignment]
    cos = F.cosine_similarity(ref[:m], aligned, dim=1)
    return float(cos.clamp(min=0.0).mean().item())


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


def _align_dictionary(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = F.normalize(reference, dim=1)
    cand = F.normalize(candidate, dim=1)
    cost = 1.0 - (ref @ cand.T)
    assignment = _hungarian(cost)
    aligned = candidate[assignment].clone()
    signs = torch.sign((reference * aligned).sum(dim=1, keepdim=True))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    aligned = aligned * signs
    return aligned, assignment, signs.squeeze(1)


def _align_codebook(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = F.normalize(reference, dim=1)
    cand = F.normalize(candidate, dim=1)
    cost = 1.0 - (ref @ cand.T)
    assignment = _hungarian(cost)
    aligned = candidate[assignment].clone()
    inverse = torch.empty_like(assignment)
    inverse[assignment] = torch.arange(assignment.numel())
    return aligned, assignment, inverse


def _fit_dictionary(
    x: torch.Tensor,
    total_features: int,
    *,
    active: int,
    iters: int,
    lam: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = _make_generator(seed)
    init = _orthogonal_matrix(x.shape[1], gen)[:total_features].clone()
    dictionary = _normalize_rows(init)
    eye = torch.eye(total_features, dtype=x.dtype, device=x.device)

    for _ in range(max(1, int(iters))):
        codes = _soft_threshold_rows(x @ dictionary.T, active)
        dictionary = torch.linalg.solve(codes.T @ codes + lam * eye, codes.T @ x)
        dictionary = _normalize_rows(dictionary)

    codes = _soft_threshold_rows(x @ dictionary.T, active)
    return dictionary, codes


def _fit_codebook(
    x: torch.Tensor,
    codebook_size: int,
    *,
    iters: int,
    seed: int,
    temp: float,
) -> torch.Tensor:
    gen = _make_generator(seed)
    indices = torch.randperm(x.shape[0], generator=gen)[:codebook_size]
    codebook = x[indices].clone()
    if codebook.shape[0] < codebook_size:
        extra = torch.randn(codebook_size - codebook.shape[0], x.shape[1], generator=gen, dtype=x.dtype)
        codebook = torch.cat([codebook, extra], dim=0)
    codebook = codebook[:codebook_size]
    codebook = codebook + 1e-6 * torch.randn(codebook.shape, generator=gen, dtype=codebook.dtype)

    for _ in range(max(1, int(iters))):
        scale = math.sqrt(float(x.shape[-1]))
        logits = (x @ codebook.T) / (scale * max(float(temp), 1e-8))
        probs = torch.softmax(logits, dim=-1)
        winners = probs.argmax(dim=-1)
        updated = []
        for atom in range(codebook_size):
            mask = winners == atom
            if mask.any():
                updated.append(x[mask].mean(dim=0))
            else:
                updated.append(codebook[atom])
        codebook = torch.stack(updated, dim=0)
    return codebook


def _fit_ridge(source: torch.Tensor, target: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([source, torch.ones(source.shape[0], 1, dtype=source.dtype, device=source.device)], dim=1)
    xtx = x.T @ x + lam * torch.eye(x.shape[1], dtype=source.dtype, device=source.device)
    xty = x.T @ target
    weight = torch.linalg.solve(xtx, xty)
    return weight[:-1], weight[-1]


def _predict_ridge(source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return source @ weight + bias


def _predict_accuracy(predicted_target: torch.Tensor, classifier: torch.Tensor, labels: torch.Tensor) -> float:
    logits = predicted_target @ classifier
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def _binary_counts(assignments: torch.Tensor, size: int) -> tuple[float, float]:
    counts = torch.bincount(assignments.view(-1), minlength=size).float()
    return _entropy_from_counts(counts), _perplexity_from_counts(counts)


def _make_true_shared_dictionary(config: ToyFeatureAtomStackBridgeConfig, *, seed: int) -> torch.Tensor:
    gen = _make_generator(seed)
    basis = _orthogonal_matrix(config.dim, gen)[: config.shared_features].clone()
    row_scale = torch.linspace(0.9, 1.35, config.shared_features, dtype=torch.float32)
    return basis * row_scale.view(-1, 1)


def _make_true_codebook(config: ToyFeatureAtomStackBridgeConfig, *, seed: int) -> torch.Tensor:
    gen = _make_generator(seed)
    family_centers = _orthogonal_matrix(config.dim, gen)[: config.route_families].clone()
    atoms: list[torch.Tensor] = []
    for family in range(config.route_families):
        center = family_centers[family]
        family_scale = 1.0 + 0.25 * family
        for atom in range(config.atoms_per_family):
            direction = center + 0.35 * torch.randn(config.dim, generator=gen, dtype=torch.float32)
            direction = direction / direction.norm().clamp_min(1e-8)
            radius = config.atom_scale * family_scale
            if atom == 0:
                radius *= 1.8
            else:
                radius *= 0.88 + 0.08 * atom
            atoms.append(direction * radius)
    return torch.stack(atoms, dim=0)


def _make_private_basis(config: ToyFeatureAtomStackBridgeConfig, *, seed: int, count: int) -> torch.Tensor:
    gen = _make_generator(seed)
    basis = _orthogonal_matrix(config.dim, gen)[: count].clone()
    scales = torch.linspace(0.4, 0.85, count, dtype=torch.float32)
    return basis * scales.view(-1, 1)


def _apply_row_gauge(dictionary: torch.Tensor, *, seed: int) -> torch.Tensor:
    gen = _make_generator(seed)
    perm = torch.randperm(dictionary.shape[0], generator=gen)
    signed = dictionary[perm].clone()
    flips = torch.where(
        torch.randn(dictionary.shape[0], generator=gen) >= 0,
        torch.ones(dictionary.shape[0], dtype=torch.float32),
        -torch.ones(dictionary.shape[0], dtype=torch.float32),
    )
    return signed * flips.view(-1, 1)


def _sample_split(
    config: ToyFeatureAtomStackBridgeConfig,
    *,
    seed: int,
    examples: int,
    source_shared: torch.Tensor,
    target_shared: torch.Tensor,
    source_atoms: torch.Tensor,
    target_atoms: torch.Tensor,
    source_private_basis: torch.Tensor,
    target_private_basis: torch.Tensor,
    classifier: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(seed)
    shared_active = max(1, min(config.shared_features, int(round(config.shared_sparsity * config.shared_features))))
    private_active = max(1, min(source_private_basis.shape[0], int(round(config.private_sparsity * source_private_basis.shape[0]))))
    atom_id = torch.randint(source_atoms.shape[0], (examples,), generator=gen)
    shared_codes = torch.zeros(examples, config.shared_features, dtype=torch.float32)
    source_private_codes = torch.zeros(examples, source_private_basis.shape[0], dtype=torch.float32)
    target_private_codes = torch.zeros(examples, target_private_basis.shape[0], dtype=torch.float32)

    for row in range(examples):
        shared_idx = torch.randperm(config.shared_features, generator=gen)[:shared_active]
        source_private_idx = torch.randperm(source_private_basis.shape[0], generator=gen)[:private_active]
        target_private_idx = torch.randperm(target_private_basis.shape[0], generator=gen)[:private_active]
        shared_codes[row, shared_idx] = config.shared_scale * torch.randn(shared_idx.numel(), generator=gen, dtype=torch.float32)
        source_private_codes[row, source_private_idx] = config.private_scale * torch.randn(
            source_private_idx.numel(), generator=gen, dtype=torch.float32
        )
        target_private_codes[row, target_private_idx] = config.private_scale * torch.randn(
            target_private_idx.numel(), generator=gen, dtype=torch.float32
        )

    atom_amp = config.atom_scale + 0.30 * torch.randn(examples, generator=gen, dtype=torch.float32)
    source = (
        shared_codes @ source_shared
        + atom_amp.unsqueeze(-1) * source_atoms[atom_id]
        + source_private_codes @ source_private_basis
        + config.noise * torch.randn(examples, config.dim, generator=gen, dtype=torch.float32)
    )
    target = (
        shared_codes @ target_shared
        + atom_amp.unsqueeze(-1) * target_atoms[atom_id]
        + target_private_codes @ target_private_basis
        + config.noise * torch.randn(examples, config.dim, generator=gen, dtype=torch.float32)
    )
    logits = target @ classifier + 0.18 * shared_codes[:, :1] + 0.12 * atom_amp.unsqueeze(-1)
    labels = logits.argmax(dim=-1)
    return {
        "source": source.float(),
        "target": target.float(),
        "shared_codes": shared_codes.float(),
        "atom_id": atom_id.long(),
        "atom_amp": atom_amp.float(),
        "labels": labels.long(),
    }


def _build_problem(config: ToyFeatureAtomStackBridgeConfig) -> dict[str, torch.Tensor]:
    total_atoms = config.route_families * config.atoms_per_family
    true_shared = _make_true_shared_dictionary(config, seed=config.seed + 11)
    true_atoms = _make_true_codebook(config, seed=config.seed + 17)
    source_shared = _apply_row_gauge(true_shared, seed=config.seed + 23)
    target_shared = _apply_row_gauge(true_shared, seed=config.seed + 29)
    source_atoms = _apply_row_gauge(true_atoms, seed=config.seed + 31)
    target_atoms = _apply_row_gauge(true_atoms, seed=config.seed + 37)
    source_private_basis = _make_private_basis(config, seed=config.seed + 41, count=config.source_private_features)
    target_private_basis = _make_private_basis(config, seed=config.seed + 47, count=config.target_private_features)
    classifier = torch.randn(config.dim, config.classes, generator=_make_generator(config.seed + 53), dtype=torch.float32)
    return {
        "total_atoms": total_atoms,
        "true_shared": true_shared,
        "true_atoms": true_atoms,
        "source_shared": source_shared,
        "target_shared": target_shared,
        "source_atoms": source_atoms,
        "target_atoms": target_atoms,
        "source_private_basis": source_private_basis,
        "target_private_basis": target_private_basis,
        "classifier": classifier,
        "train": _sample_split(
            config,
            seed=config.seed + 101,
            examples=config.train_examples,
            source_shared=source_shared,
            target_shared=target_shared,
            source_atoms=source_atoms,
            target_atoms=target_atoms,
            source_private_basis=source_private_basis,
            target_private_basis=target_private_basis,
            classifier=classifier,
        ),
        "test": _sample_split(
            config,
            seed=config.seed + 203,
            examples=config.test_examples,
            source_shared=source_shared,
            target_shared=target_shared,
            source_atoms=source_atoms,
            target_atoms=target_atoms,
            source_private_basis=source_private_basis,
            target_private_basis=target_private_basis,
            classifier=classifier,
        ),
    }


def _project_latent_usage(
    x: torch.Tensor,
    shared_dictionary: torch.Tensor,
    atom_codebook: torch.Tensor,
) -> dict[str, float]:
    shared_scores = x @ shared_dictionary.T
    atom_scores = x @ atom_codebook.T
    shared_winners = shared_scores.abs().argmax(dim=-1)
    atom_winners = atom_scores.abs().argmax(dim=-1)
    shared_entropy, shared_perplexity = _binary_counts(shared_winners, shared_dictionary.shape[0])
    atom_entropy, atom_perplexity = _binary_counts(atom_winners, atom_codebook.shape[0])
    return {
        "shared_entropy": shared_entropy,
        "shared_perplexity": shared_perplexity,
        "atom_entropy": atom_entropy,
        "atom_perplexity": atom_perplexity,
    }


def _feature_reconstruction_metrics(
    predicted_target: torch.Tensor,
    target_shared: torch.Tensor,
    target_atoms: torch.Tensor,
    *,
    shared_features: int,
    total_atoms: int,
    active_shared: int,
) -> dict[str, float]:
    shared_proj = predicted_target @ target_shared.T
    shared_idx = torch.topk(shared_proj.abs(), k=min(active_shared, shared_features), dim=1).indices
    shared_support = torch.zeros_like(shared_proj, dtype=torch.bool)
    shared_support.scatter_(1, shared_idx, True)
    true_shared_support = torch.ones(predicted_target.shape[0], shared_features, dtype=torch.bool, device=predicted_target.device)
    if active_shared < shared_features:
        true_shared_support = true_shared_support
    tp = (shared_support & true_shared_support).sum(dim=1).float()
    fp = ((~true_shared_support) & shared_support).sum(dim=1).float()
    fn = (true_shared_support & (~shared_support)).sum(dim=1).float()
    precision = (tp / (tp + fp).clamp_min(1.0)).mean().item()
    recall = (tp / (tp + fn).clamp_min(1.0)).mean().item()
    shared_f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    atom_scores = predicted_target @ target_atoms.T
    atom_winners = atom_scores.argmax(dim=-1)
    atom_counts = torch.bincount(atom_winners, minlength=total_atoms).float()
    atom_entropy = _entropy_from_counts(atom_counts)
    atom_perplexity = _perplexity_from_counts(atom_counts)
    return {
        "shared_feature_recovery": float(shared_f1),
        "atom_entropy": atom_entropy,
        "atom_perplexity": atom_perplexity,
    }


def _stacked_bytes_proxy(
    *,
    shared_features: int,
    total_atoms: int,
    dim: int,
    protected_shared: int = 0,
    protected_atoms: int = 0,
) -> float:
    shared_bytes = 2 * _bytes_for_matrix(shared_features, dim) + _bytes_for_matrix(shared_features, shared_features)
    atom_bytes = 2 * _bytes_for_matrix(total_atoms, dim) + _bytes_for_matrix(total_atoms, total_atoms)
    bridge_bytes = _bytes_for_matrix(shared_features + total_atoms, shared_features + total_atoms)
    protected_bytes = _bytes_for_values(protected_shared * dim, 16.0) + _bytes_for_values(protected_atoms * dim, 16.0)
    return float(shared_bytes + atom_bytes + bridge_bytes + protected_bytes)


def _compute_proxy(
    *,
    examples: int,
    dim: int,
    shared_features: int,
    total_atoms: int,
    shared_iters: int,
    atom_iters: int,
    stacked: bool,
    protected: bool = False,
) -> float:
    base = examples * dim * (shared_features * max(1, shared_iters) + total_atoms * max(1, atom_iters))
    if stacked:
        base *= 1.25
    if protected:
        base *= 1.10
    return float(base)


def _evaluate_method(config: ToyFeatureAtomStackBridgeConfig, problem: dict[str, Any], method: str) -> dict[str, Any]:
    train = problem["train"]
    test = problem["test"]
    classifier = problem["classifier"]
    total_atoms = problem["total_atoms"]
    active_shared = max(1, min(config.shared_features, int(round(config.shared_sparsity * config.shared_features))))
    raw_weight, raw_bias = _fit_ridge(train["source"], train["target"], config.bridge_lam)
    raw_train_pred = _predict_ridge(train["source"], raw_weight, raw_bias)
    raw_test_pred = _predict_ridge(test["source"], raw_weight, raw_bias)
    raw_usage = _project_latent_usage(raw_test_pred, problem["true_shared"], problem["true_atoms"])

    if method == "raw_ridge":
        row = {
            "method": method,
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "route_families": int(config.route_families),
            "atoms_per_family": int(config.atoms_per_family),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "train_mse": float(F.mse_loss(raw_train_pred, train["target"]).item()),
            "test_mse": float(F.mse_loss(raw_test_pred, test["target"]).item()),
            "train_accuracy": _predict_accuracy(raw_train_pred, classifier, train["labels"]),
            "test_accuracy": _predict_accuracy(raw_test_pred, classifier, test["labels"]),
            "shared_feature_recovery": 0.0,
            "atom_recovery": 0.0,
            "shared_entropy": raw_usage["shared_entropy"],
            "shared_perplexity": raw_usage["shared_perplexity"],
            "atom_entropy": raw_usage["atom_entropy"],
            "atom_perplexity": raw_usage["atom_perplexity"],
            "bytes_proxy": float(_bytes_for_matrix(config.dim, config.dim)),
            "compute_proxy": float(config.train_examples * config.dim * config.dim),
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
        return row

    if method == "shared_feature_only":
        source_dict, source_codes = _fit_dictionary(
            train["source"],
            config.shared_features,
            active=active_shared,
            iters=config.shared_iters,
            lam=config.dictionary_lam,
            seed=config.seed + 301,
        )
        target_dict, target_codes = _fit_dictionary(
            train["target"],
            config.shared_features,
            active=active_shared,
            iters=config.shared_iters,
            lam=config.dictionary_lam,
            seed=config.seed + 303,
        )
        aligned_target_dict, assignment, signs = _align_dictionary(source_dict, target_dict)
        aligned_target_codes = target_codes[:, assignment] * signs.view(1, -1)
        bridge_w, bridge_b = _fit_ridge(source_codes, aligned_target_codes, config.bridge_lam)
        test_source_codes = _soft_threshold_rows(test["source"] @ source_dict.T, active_shared)
        pred_target_codes = _predict_ridge(test_source_codes, bridge_w, bridge_b)
        pred_target = pred_target_codes @ aligned_target_dict
        usage = _project_latent_usage(pred_target, problem["true_shared"], problem["true_atoms"])
        row = {
            "method": method,
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "route_families": int(config.route_families),
            "atoms_per_family": int(config.atoms_per_family),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "train_mse": float(F.mse_loss(source_codes @ aligned_target_dict, train["target"]).item()),
            "test_mse": float(F.mse_loss(pred_target, test["target"]).item()),
            "train_accuracy": _predict_accuracy(source_codes @ aligned_target_dict, classifier, train["labels"]),
            "test_accuracy": _predict_accuracy(pred_target, classifier, test["labels"]),
            "shared_feature_recovery": 0.5 * (
                _match_recovery(problem["true_shared"], source_dict) + _match_recovery(problem["true_shared"], aligned_target_dict)
            ),
            "atom_recovery": 0.0,
            "shared_entropy": usage["shared_entropy"],
            "shared_perplexity": usage["shared_perplexity"],
            "atom_entropy": usage["atom_entropy"],
            "atom_perplexity": usage["atom_perplexity"],
            "bytes_proxy": float(
                2 * _bytes_for_matrix(config.shared_features, config.dim) + _bytes_for_matrix(config.shared_features, config.shared_features)
            ),
            "compute_proxy": _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                shared_features=config.shared_features,
                total_atoms=0,
                shared_iters=config.shared_iters,
                atom_iters=1,
                stacked=False,
            ),
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
        return row

    if method == "route_atom_only":
        source_codebook = _fit_codebook(
            train["source"],
            total_atoms,
            iters=config.atom_iters,
            seed=config.seed + 401,
            temp=config.codebook_temp,
        )
        target_codebook = _fit_codebook(
            train["target"],
            total_atoms,
            iters=config.atom_iters,
            seed=config.seed + 403,
            temp=config.codebook_temp,
        )
        aligned_target_codebook, assignment, inverse = _align_codebook(source_codebook, target_codebook)
        source_assign = (train["source"] @ source_codebook.T).argmax(dim=-1)
        target_assign = (train["target"] @ target_codebook.T).argmax(dim=-1)
        aligned_target_assign = inverse[target_assign]
        source_onehot = F.one_hot(source_assign, num_classes=total_atoms).float()
        aligned_target_onehot = F.one_hot(aligned_target_assign, num_classes=total_atoms).float()
        bridge_w, bridge_b = _fit_ridge(source_onehot, aligned_target_onehot, config.bridge_lam)
        test_assign = (test["source"] @ source_codebook.T).argmax(dim=-1)
        test_onehot = F.one_hot(test_assign, num_classes=total_atoms).float()
        pred_target_onehot = _predict_ridge(test_onehot, bridge_w, bridge_b)
        pred_target = pred_target_onehot @ aligned_target_codebook
        usage = _project_latent_usage(pred_target, problem["true_shared"], problem["true_atoms"])
        row = {
            "method": method,
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "route_families": int(config.route_families),
            "atoms_per_family": int(config.atoms_per_family),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "train_mse": float(F.mse_loss(source_onehot @ aligned_target_codebook, train["target"]).item()),
            "test_mse": float(F.mse_loss(pred_target, test["target"]).item()),
            "train_accuracy": _predict_accuracy(source_onehot @ aligned_target_codebook, classifier, train["labels"]),
            "test_accuracy": _predict_accuracy(pred_target, classifier, test["labels"]),
            "shared_feature_recovery": 0.0,
            "atom_recovery": 0.5 * (
                _match_recovery(problem["true_atoms"], source_codebook) + _match_recovery(problem["true_atoms"], aligned_target_codebook)
            ),
            "shared_entropy": usage["shared_entropy"],
            "shared_perplexity": usage["shared_perplexity"],
            "atom_entropy": usage["atom_entropy"],
            "atom_perplexity": usage["atom_perplexity"],
            "bytes_proxy": float(
                2 * _bytes_for_matrix(total_atoms, config.dim) + _bytes_for_matrix(total_atoms, total_atoms)
            ),
            "compute_proxy": _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                shared_features=0,
                total_atoms=total_atoms,
                shared_iters=1,
                atom_iters=config.atom_iters,
                stacked=False,
            ),
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
        return row

    if method in {"stacked_feature_atom", "protected_stacked_feature_atom"}:
        source_shared_dict, source_shared_codes = _fit_dictionary(
            train["source"],
            config.shared_features,
            active=active_shared,
            iters=config.shared_iters,
            lam=config.dictionary_lam,
            seed=config.seed + 501,
        )
        target_shared_dict, target_shared_codes = _fit_dictionary(
            train["target"],
            config.shared_features,
            active=active_shared,
            iters=config.shared_iters,
            lam=config.dictionary_lam,
            seed=config.seed + 503,
        )
        aligned_target_shared_dict, shared_assignment, shared_signs = _align_dictionary(source_shared_dict, target_shared_dict)
        aligned_target_shared_codes = target_shared_codes[:, shared_assignment] * shared_signs.view(1, -1)

        source_shared_recon = source_shared_codes @ source_shared_dict
        target_shared_recon = target_shared_codes @ target_shared_dict
        source_atom_residual = train["source"] - source_shared_recon
        target_atom_residual = train["target"] - target_shared_recon
        source_codebook = _fit_codebook(
            source_atom_residual,
            total_atoms,
            iters=config.atom_iters,
            seed=config.seed + 601,
            temp=config.codebook_temp,
        )
        target_codebook = _fit_codebook(
            target_atom_residual,
            total_atoms,
            iters=config.atom_iters,
            seed=config.seed + 603,
            temp=config.codebook_temp,
        )
        aligned_target_codebook, atom_assignment, atom_inverse = _align_codebook(source_codebook, target_codebook)

        source_atom_assign = (source_atom_residual @ source_codebook.T).argmax(dim=-1)
        target_atom_assign = (target_atom_residual @ target_codebook.T).argmax(dim=-1)
        aligned_target_atom_assign = atom_inverse[target_atom_assign]

        source_latent = torch.cat(
            [
                source_shared_codes,
                F.one_hot(source_atom_assign, num_classes=total_atoms).float(),
            ],
            dim=1,
        )
        target_latent = torch.cat(
            [
                aligned_target_shared_codes,
                F.one_hot(aligned_target_atom_assign, num_classes=total_atoms).float(),
            ],
            dim=1,
        )
        bridge_w, bridge_b = _fit_ridge(source_latent, target_latent, config.bridge_lam)

        test_shared_codes = _soft_threshold_rows(test["source"] @ source_shared_dict.T, active_shared)
        test_shared_latent = test_shared_codes
        test_shared_recon = test_shared_codes @ source_shared_dict
        test_atom_residual = test["source"] - test_shared_recon
        test_atom_assign = (test_atom_residual @ source_codebook.T).argmax(dim=-1)
        test_latent = torch.cat(
            [
                test_shared_latent,
                F.one_hot(test_atom_assign, num_classes=total_atoms).float(),
            ],
            dim=1,
        )
        pred_target_latent = _predict_ridge(test_latent, bridge_w, bridge_b)
        pred_shared_latent = pred_target_latent[:, : config.shared_features]
        pred_atom_latent = pred_target_latent[:, config.shared_features :]
        pred_target = pred_shared_latent @ aligned_target_shared_dict + pred_atom_latent @ aligned_target_codebook
        usage = _project_latent_usage(pred_target, problem["true_shared"], problem["true_atoms"])

        shared_recovery = 0.5 * (
            _match_recovery(problem["true_shared"], source_shared_dict)
            + _match_recovery(problem["true_shared"], aligned_target_shared_dict)
        )
        atom_recovery = 0.5 * (
            _match_recovery(problem["true_atoms"], source_codebook)
            + _match_recovery(problem["true_atoms"], aligned_target_codebook)
        )

        protected_shared = min(config.protected_shared, config.shared_features)
        protected_atoms = min(config.protected_atoms, total_atoms)
        if method == "protected_stacked_feature_atom":
            shared_norm_order = source_shared_dict.norm(dim=1).argsort(descending=True)
            atom_norm_order = source_codebook.norm(dim=1).argsort(descending=True)
            shared_keep = set(shared_norm_order[:protected_shared].tolist())
            atom_keep = set(atom_norm_order[:protected_atoms].tolist())
            quant_source_shared = source_shared_dict.clone()
            quant_target_shared = aligned_target_shared_dict.clone()
            quant_source_atoms = source_codebook.clone()
            quant_target_atoms = aligned_target_codebook.clone()
            for idx in range(config.shared_features):
                if idx not in shared_keep:
                    quant_source_shared[idx] = torch.round(quant_source_shared[idx] * 128.0) / 128.0
                    quant_target_shared[idx] = torch.round(quant_target_shared[idx] * 128.0) / 128.0
            for idx in range(total_atoms):
                if idx not in atom_keep:
                    quant_source_atoms[idx] = torch.round(quant_source_atoms[idx] * 128.0) / 128.0
                    quant_target_atoms[idx] = torch.round(quant_target_atoms[idx] * 128.0) / 128.0
            pred_target = pred_shared_latent @ quant_target_shared + pred_atom_latent @ quant_target_atoms
            usage = _project_latent_usage(pred_target, problem["true_shared"], problem["true_atoms"])
            bytes_proxy = _stacked_bytes_proxy(
                shared_features=config.shared_features,
                total_atoms=total_atoms,
                dim=config.dim,
                protected_shared=protected_shared,
                protected_atoms=protected_atoms,
            )
            compute_proxy = _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                shared_features=config.shared_features,
                total_atoms=total_atoms,
                shared_iters=config.shared_iters,
                atom_iters=config.atom_iters,
                stacked=True,
                protected=True,
            )
        else:
            bytes_proxy = _stacked_bytes_proxy(
                shared_features=config.shared_features,
                total_atoms=total_atoms,
                dim=config.dim,
            )
            compute_proxy = _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                shared_features=config.shared_features,
                total_atoms=total_atoms,
                shared_iters=config.shared_iters,
                atom_iters=config.atom_iters,
                stacked=True,
            )

        row = {
            "method": method,
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "route_families": int(config.route_families),
            "atoms_per_family": int(config.atoms_per_family),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "train_mse": float(F.mse_loss(
                source_latent[:, : config.shared_features] @ aligned_target_shared_dict + source_latent[:, config.shared_features :] @ aligned_target_codebook,
                train["target"],
            ).item()),
            "test_mse": float(F.mse_loss(pred_target, test["target"]).item()),
            "train_accuracy": _predict_accuracy(
                source_latent[:, : config.shared_features] @ aligned_target_shared_dict + source_latent[:, config.shared_features :] @ aligned_target_codebook,
                classifier,
                train["labels"],
            ),
            "test_accuracy": _predict_accuracy(pred_target, classifier, test["labels"]),
            "shared_feature_recovery": shared_recovery,
            "atom_recovery": atom_recovery,
            "shared_entropy": usage["shared_entropy"],
            "shared_perplexity": usage["shared_perplexity"],
            "atom_entropy": usage["atom_entropy"],
            "atom_perplexity": usage["atom_perplexity"],
            "bytes_proxy": float(bytes_proxy),
            "compute_proxy": float(compute_proxy),
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
        return row

    if method == "oracle":
        pred_target = test["target"]
        usage = _project_latent_usage(pred_target, problem["true_shared"], problem["true_atoms"])
        row = {
            "method": method,
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "route_families": int(config.route_families),
            "atoms_per_family": int(config.atoms_per_family),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "train_mse": 0.0,
            "test_mse": 0.0,
            "train_accuracy": _predict_accuracy(train["target"], classifier, train["labels"]),
            "test_accuracy": 1.0,
            "shared_feature_recovery": 1.0,
            "atom_recovery": 1.0,
            "shared_entropy": usage["shared_entropy"],
            "shared_perplexity": usage["shared_perplexity"],
            "atom_entropy": usage["atom_entropy"],
            "atom_perplexity": usage["atom_perplexity"],
            "bytes_proxy": 0.0,
            "compute_proxy": 0.0,
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
        return row

    raise ValueError(f"Unknown method: {method}")


def run_experiment(config: ToyFeatureAtomStackBridgeConfig) -> list[dict[str, Any]]:
    problem = _build_problem(config)
    rows = [_evaluate_method(config, problem, method) for method in METHODS]
    raw_accuracy = rows[0]["test_accuracy"]
    raw_mse = rows[0]["test_mse"]
    for row in rows[1:]:
        row["accuracy_delta_vs_raw"] = float(row["test_accuracy"] - raw_accuracy)
        row["mse_delta_vs_raw"] = float(row["test_mse"] - raw_mse)
        row["help_vs_raw"] = float(max(0.0, row["accuracy_delta_vs_raw"]))
        row["harm_vs_raw"] = float(max(0.0, -row["accuracy_delta_vs_raw"]))
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"


def write_markdown_summary(config: ToyFeatureAtomStackBridgeConfig, rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    lines = [
        "# Toy Feature Atom Stack Bridge",
        "",
        f"- Seed: `{config.seed}`",
        f"- Train examples: `{config.train_examples}`",
        f"- Test examples: `{config.test_examples}`",
        f"- Shared features: `{config.shared_features}`",
        f"- Route families: `{config.route_families}`",
        f"- Atoms per family: `{config.atoms_per_family}`",
        f"- Protected shared rows: `{config.protected_shared}`",
        f"- Protected atoms: `{config.protected_atoms}`",
        "",
        "| Method | Test Acc | Test MSE | Shared rec | Atom rec | Shared H | Atom H | Bytes proxy | Compute proxy | Help vs raw | Harm vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {test_accuracy} | {test_mse} | {shared_feature_recovery} | {atom_recovery} | {shared_entropy} | {atom_entropy} | {bytes_proxy} | {compute_proxy} | {help_vs_raw} | {harm_vs_raw} |".format(
                method=row["method"],
                test_accuracy=_fmt(row["test_accuracy"]),
                test_mse=_fmt(row["test_mse"]),
                shared_feature_recovery=_fmt(row["shared_feature_recovery"]),
                atom_recovery=_fmt(row["atom_recovery"]),
                shared_entropy=_fmt(row["shared_entropy"]),
                atom_entropy=_fmt(row["atom_entropy"]),
                bytes_proxy=_fmt(row["bytes_proxy"]),
                compute_proxy=_fmt(row["compute_proxy"]),
                help_vs_raw=_fmt(row["help_vs_raw"]),
                harm_vs_raw=_fmt(row["harm_vs_raw"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy feature+atom stack bridge ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=144)
    parser.add_argument("--test-examples", type=int, default=96)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--shared-features", type=int, default=6)
    parser.add_argument("--route-families", type=int, default=4)
    parser.add_argument("--atoms-per-family", type=int, default=4)
    parser.add_argument("--source-private-features", type=int, default=4)
    parser.add_argument("--target-private-features", type=int, default=4)
    parser.add_argument("--shared-sparsity", type=float, default=0.35)
    parser.add_argument("--private-sparsity", type=float, default=0.20)
    parser.add_argument("--shared-scale", type=float, default=1.15)
    parser.add_argument("--atom-scale", type=float, default=1.8)
    parser.add_argument("--private-scale", type=float, default=0.45)
    parser.add_argument("--noise", type=float, default=0.04)
    parser.add_argument("--shared-iters", type=int, default=8)
    parser.add_argument("--atom-iters", type=int, default=8)
    parser.add_argument("--bridge-lam", type=float, default=1e-2)
    parser.add_argument("--dictionary-lam", type=float, default=1e-3)
    parser.add_argument("--codebook-temp", type=float, default=0.35)
    parser.add_argument("--protected-shared", type=int, default=2)
    parser.add_argument("--protected-atoms", type=int, default=3)
    parser.add_argument("--classes", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyFeatureAtomStackBridgeConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        shared_features=args.shared_features,
        route_families=args.route_families,
        atoms_per_family=args.atoms_per_family,
        source_private_features=args.source_private_features,
        target_private_features=args.target_private_features,
        shared_sparsity=args.shared_sparsity,
        private_sparsity=args.private_sparsity,
        shared_scale=args.shared_scale,
        atom_scale=args.atom_scale,
        private_scale=args.private_scale,
        noise=args.noise,
        shared_iters=args.shared_iters,
        atom_iters=args.atom_iters,
        bridge_lam=args.bridge_lam,
        dictionary_lam=args.dictionary_lam,
        codebook_temp=args.codebook_temp,
        protected_shared=args.protected_shared,
        protected_atoms=args.protected_atoms,
        classes=args.classes,
    )
    rows = run_experiment(config)
    payload = {"config": asdict(config), "methods": list(METHODS), "rows": rows}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path = pathlib.Path(args.output_md) if args.output_md else output.with_suffix(".md")
    write_markdown_summary(config, rows, markdown_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
