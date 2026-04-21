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
    "raw_residual_bridge",
    "separate_per_model_dictionaries",
    "shared_dictionary_crosscoder",
    "symmetry_aware_shared_dictionary",
    "oracle_upper_bound",
)


@dataclass(frozen=True)
class ToySharedFeatureDictionaryConfig:
    seed: int = 0
    train_examples: int = 96
    test_examples: int = 96
    dim: int = 16
    shared_features: int = 6
    source_private_features: int = 4
    target_private_features: int = 4
    sparsity: float = 0.25
    noise: float = 0.03
    dictionary_iters: int = 8
    dictionary_lam: float = 1e-3
    bridge_lam: float = 1e-2


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float32))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _row_permutation(size: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(size, generator=generator)


def _make_canonical_dictionary(
    total_features: int,
    dim: int,
    *,
    seed: int,
) -> torch.Tensor:
    gen = _make_generator(seed)
    basis = _orthogonal_matrix(dim, gen)
    if total_features > dim:
        raise ValueError("feature count must not exceed the ambient dimension")
    rows = basis[:total_features].clone()
    row_scale = torch.linspace(0.9, 1.3, total_features, dtype=torch.float32)
    rows = rows * row_scale.view(-1, 1)
    return rows


def _apply_row_gauge(
    dictionary: torch.Tensor,
    *,
    seed: int,
    permutation_strength: float = 1.0,
) -> torch.Tensor:
    dim = dictionary.shape[0]
    gen = _make_generator(seed)
    perm = _row_permutation(dim, gen)
    if permutation_strength < 1.0 and dim > 1:
        keep = max(1, int(round(permutation_strength * dim)))
        perm = torch.cat([perm[:keep], torch.arange(keep, dim, dtype=torch.long)])
    signed = dictionary[perm].clone()
    flip_mask = torch.where(
        torch.randn(dim, generator=gen) >= 0,
        torch.ones(dim, dtype=torch.float32),
        -torch.ones(dim, dtype=torch.float32),
    )
    signed = signed * flip_mask.view(-1, 1)
    return signed


def _make_sparse_codes(
    count: int,
    total_features: int,
    shared_features: int,
    source_private_features: int,
    target_private_features: int,
    sparsity: float,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = _make_generator(seed)
    active = max(1, min(total_features, int(round(float(sparsity) * total_features))))
    active_shared = max(1, min(shared_features, int(round(active * 0.5))))
    active_source_private = max(1, min(source_private_features, max(active - active_shared, 1)))
    active_target_private = max(1, min(target_private_features, max(active - active_shared, 1)))

    shared = torch.zeros(count, shared_features, dtype=torch.float32)
    source_private = torch.zeros(count, source_private_features, dtype=torch.float32)
    target_private = torch.zeros(count, target_private_features, dtype=torch.float32)

    for row in range(count):
        shared_idx = torch.randperm(shared_features, generator=gen)[:active_shared]
        source_idx = torch.randperm(source_private_features, generator=gen)[:active_source_private]
        target_idx = torch.randperm(target_private_features, generator=gen)[:active_target_private]
        shared[row, shared_idx] = torch.randn(shared_idx.numel(), generator=gen, dtype=torch.float32)
        source_private[row, source_idx] = 0.85 * torch.randn(
            source_idx.numel(), generator=gen, dtype=torch.float32
        )
        target_private[row, target_idx] = 0.85 * torch.randn(
            target_idx.numel(), generator=gen, dtype=torch.float32
        )

    source = torch.cat(
        [
            shared,
            source_private,
            torch.zeros(count, target_private_features, dtype=torch.float32),
        ],
        dim=1,
    )
    target = torch.cat(
        [
            shared,
            torch.zeros(count, source_private_features, dtype=torch.float32),
            target_private,
        ],
        dim=1,
    )
    return source, target, shared, torch.cat([shared, source_private, target_private], dim=1)


def _add_noise(x: torch.Tensor, noise: float, *, seed: int) -> torch.Tensor:
    if noise <= 0:
        return x
    gen = _make_generator(seed)
    return x + noise * torch.randn(x.shape, generator=gen, dtype=x.dtype)


def _build_problem(config: ToySharedFeatureDictionaryConfig, *, split: str) -> dict[str, torch.Tensor]:
    if split not in {"train", "test"}:
        raise ValueError(f"unknown split: {split}")

    count = config.train_examples if split == "train" else config.test_examples
    total_features = config.shared_features + config.source_private_features + config.target_private_features
    if total_features > config.dim:
        raise ValueError("feature count exceeds ambient dimension")

    offset = 0 if split == "train" else 100_000
    seed_base = config.seed + offset
    canonical = _make_canonical_dictionary(total_features, config.dim, seed=seed_base + 11)
    source_gauge = _orthogonal_matrix(config.dim, _make_generator(seed_base + 23))
    target_gauge = _orthogonal_matrix(config.dim, _make_generator(seed_base + 31))
    source_perm = _apply_row_gauge(canonical, seed=seed_base + 41)
    target_perm = _apply_row_gauge(canonical, seed=seed_base + 53)

    source_decoder = source_perm @ source_gauge
    target_decoder = target_perm @ target_gauge

    source_codes, target_codes, shared_codes, full_codes = _make_sparse_codes(
        count,
        total_features,
        config.shared_features,
        config.source_private_features,
        config.target_private_features,
        config.sparsity,
        seed=seed_base + 67,
    )
    source = _add_noise(source_codes @ source_decoder, config.noise, seed=seed_base + 71)
    target = _add_noise(target_codes @ target_decoder, config.noise, seed=seed_base + 73)

    return {
        "source": source,
        "target": target,
        "shared_codes": shared_codes,
        "full_codes": full_codes,
        "source_decoder": source_decoder,
        "target_decoder": target_decoder,
        "source_codes": source_codes,
        "target_codes": target_codes,
        "canonical_dictionary": canonical,
    }


def _soft_threshold_rows(x: torch.Tensor, active: int) -> torch.Tensor:
    if active >= x.shape[1]:
        return x
    values, indices = torch.topk(x.abs(), k=active, dim=1)
    del values
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return torch.where(mask, x, torch.zeros_like(x))


def _normalize_dictionary_rows(dictionary: torch.Tensor) -> torch.Tensor:
    normalized = dictionary.clone()
    norms = normalized.norm(dim=1, keepdim=True).clamp_min(1e-8)
    normalized = normalized / norms
    max_idx = normalized.abs().argmax(dim=1, keepdim=True)
    signs = torch.gather(normalized, 1, max_idx).sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return normalized * signs


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
    dictionary = _normalize_dictionary_rows(init)
    codes = torch.zeros(x.shape[0], total_features, dtype=x.dtype)

    eye = torch.eye(total_features, dtype=x.dtype, device=x.device)
    for _ in range(max(1, int(iters))):
        codes = _soft_threshold_rows(x @ dictionary.T, active)
        gram = codes.T @ codes + lam * eye
        dictionary = torch.linalg.solve(gram, codes.T @ x)
        dictionary = _normalize_dictionary_rows(dictionary)
    codes = _soft_threshold_rows(x @ dictionary.T, active)
    return dictionary, codes


def _fit_ridge(source_train: torch.Tensor, target_train: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([source_train, torch.ones(source_train.shape[0], 1, dtype=source_train.dtype)], dim=1)
    xtx = x.T @ x + lam * torch.eye(x.shape[1], dtype=source_train.dtype)
    xty = x.T @ target_train
    weight = torch.linalg.solve(xtx, xty)
    return weight[:-1], weight[-1]


def _predict_ridge(source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return source @ weight + bias


def _align_rows(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, float]:
    ref = F.normalize(reference, dim=1)
    cand = F.normalize(candidate, dim=1)
    cost = 1.0 - (ref @ cand.T)
    assignment = _hungarian(cost)
    aligned = candidate[assignment].clone()
    signs = torch.sign((reference * aligned).sum(dim=1, keepdim=True))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    aligned = aligned * signs
    residual = float((1.0 - F.cosine_similarity(reference, aligned, dim=1)).mean().item())
    return aligned, residual


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


def _infer_codes(x: torch.Tensor, dictionary: torch.Tensor, active: int) -> torch.Tensor:
    return _soft_threshold_rows(x @ dictionary.T, active)


def _target_projection_metrics(
    predicted_target: torch.Tensor,
    target_decoder: torch.Tensor,
    shared_features: int,
    true_shared_codes: torch.Tensor,
    *,
    active: int,
) -> dict[str, float]:
    projected = predicted_target @ target_decoder.T
    shared_pred = projected[:, :shared_features]
    shared_active = max(1, min(shared_features, active))
    _, shared_indices = torch.topk(shared_pred.abs(), k=shared_active, dim=1)
    support_true = true_shared_codes.abs() > 1e-8
    support_pred = torch.zeros_like(shared_pred, dtype=torch.bool)
    support_pred.scatter_(1, shared_indices, True)
    tp = (support_true & support_pred).sum(dim=1).float()
    fp = ((~support_true) & support_pred).sum(dim=1).float()
    fn = (support_true & (~support_pred)).sum(dim=1).float()
    precision = (tp / (tp + fp).clamp_min(1.0)).mean().item()
    recall = (tp / (tp + fn).clamp_min(1.0)).mean().item()
    f1 = (2 * precision * recall / max(precision + recall, 1e-8))
    topk_values, _ = torch.topk(projected.abs(), k=active, dim=1)
    sparsity = float((topk_values.sum(dim=1) / projected.abs().sum(dim=1).clamp_min(1e-8)).mean().item())
    return {
        "shared_feature_recovery": float(f1),
        "sparsity": sparsity,
    }


def _probe_accuracy(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    x = torch.cat([train_x, torch.ones(train_x.shape[0], 1, dtype=train_x.dtype)], dim=1)
    xtx = x.T @ x + 1e-3 * torch.eye(x.shape[1], dtype=train_x.dtype)
    xty = x.T @ train_y.float().unsqueeze(1)
    weight = torch.linalg.solve(xtx, xty).squeeze(1)
    logits = torch.cat([test_x, torch.ones(test_x.shape[0], 1, dtype=test_x.dtype)], dim=1) @ weight
    preds = (logits >= 0.5).long()
    return float((preds == test_y.long()).float().mean().item())


def _binary_labels(shared_codes: torch.Tensor, *, seed: int) -> torch.Tensor:
    gen = _make_generator(seed)
    weights = torch.randn(shared_codes.shape[1], generator=gen, dtype=shared_codes.dtype)
    logits = shared_codes @ weights
    logits = logits + 0.35 * shared_codes[:, 0] - 0.15 * shared_codes[:, -1]
    return (logits >= 0).long()


def _bytes_for_matrix(rows: int, cols: int) -> int:
    return int(rows * cols * 4)


def _compute_proxy(
    *,
    examples: int,
    dim: int,
    features: int,
    iters: int,
    paired: bool,
) -> float:
    multiplier = 2.0 if paired else 1.0
    return float(multiplier * examples * dim * features * max(1, iters))


def run_experiment(
    config: ToySharedFeatureDictionaryConfig,
) -> list[dict[str, Any]]:
    train = _build_problem(config, split="train")
    test = _build_problem(config, split="test")
    total_features = config.shared_features + config.source_private_features + config.target_private_features
    active = max(1, min(total_features, int(round(config.sparsity * total_features))))

    rows: list[dict[str, Any]] = []

    raw_weight, raw_bias = _fit_ridge(train["source"], train["target"], config.bridge_lam)
    raw_train_pred = _predict_ridge(train["source"], raw_weight, raw_bias)
    raw_test_pred = _predict_ridge(test["source"], raw_weight, raw_bias)
    raw_projection_metrics = _target_projection_metrics(
        raw_test_pred,
        test["target_decoder"],
        config.shared_features,
        test["shared_codes"],
        active=active,
    )
    raw_row = {
        "method": "raw_residual_bridge",
        "seed": int(config.seed),
        "dim": int(config.dim),
        "train_examples": int(config.train_examples),
        "test_examples": int(config.test_examples),
        "shared_features": int(config.shared_features),
        "source_private_features": int(config.source_private_features),
        "target_private_features": int(config.target_private_features),
        "sparsity": float(config.sparsity),
        "noise": float(config.noise),
        "train_mse": float(F.mse_loss(raw_train_pred, train["target"]).item()),
        "test_mse": float(F.mse_loss(raw_test_pred, test["target"]).item()),
        "train_accuracy": _probe_accuracy(train["target"], _binary_labels(train["shared_codes"], seed=config.seed + 5), train["target"], _binary_labels(train["shared_codes"], seed=config.seed + 5)),
        "test_accuracy": _probe_accuracy(train["target"], _binary_labels(train["shared_codes"], seed=config.seed + 5), raw_test_pred, _binary_labels(test["shared_codes"], seed=config.seed + 5)),
        "shared_feature_recovery": raw_projection_metrics["shared_feature_recovery"],
        "sparsity_rate": raw_projection_metrics["sparsity"],
        "dictionary_alignment_residual": None,
        "bytes_proxy": float(_bytes_for_matrix(config.dim, config.dim) + 4 * config.dim),
        "compute_proxy": float(_compute_proxy(
            examples=config.train_examples,
            dim=config.dim,
            features=config.dim,
            iters=1,
            paired=False,
        )),
        "accuracy_delta_vs_raw": 0.0,
        "mse_delta_vs_raw": 0.0,
        "help_vs_raw": 0.0,
        "harm_vs_raw": 0.0,
    }
    rows.append(raw_row)

    # Separate per-model dictionaries.
    source_dict, source_codes = _fit_dictionary(
        train["source"],
        total_features,
        active=active,
        iters=config.dictionary_iters,
        lam=config.dictionary_lam,
        seed=config.seed + 101,
    )
    target_dict, target_codes = _fit_dictionary(
        train["target"],
        total_features,
        active=active,
        iters=config.dictionary_iters,
        lam=config.dictionary_lam,
        seed=config.seed + 103,
    )
    code_bridge, code_bias = _fit_ridge(source_codes, target_codes, config.bridge_lam)
    separate_train_target = source_codes @ code_bridge + code_bias
    separate_test_codes = _infer_codes(test["source"], source_dict, active)
    separate_test_target = separate_test_codes @ code_bridge + code_bias
    separate_train_pred = separate_train_target @ target_dict
    separate_test_pred = separate_test_target @ target_dict
    _, residual_sep = _align_rows(source_dict, target_dict)
    metrics_sep = _target_projection_metrics(
        separate_test_pred,
        test["target_decoder"],
        config.shared_features,
        test["shared_codes"],
        active=active,
    )
    rows.append(
        {
            "method": "separate_per_model_dictionaries",
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "sparsity": float(config.sparsity),
            "noise": float(config.noise),
            "train_mse": float(F.mse_loss(separate_train_pred, train["target"]).item()),
            "test_mse": float(F.mse_loss(separate_test_pred, test["target"]).item()),
            "train_accuracy": _probe_accuracy(
                train["target"],
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
                separate_train_pred,
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
            ),
            "test_accuracy": _probe_accuracy(
                train["target"],
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
                separate_test_pred,
                _binary_labels(test["shared_codes"], seed=config.seed + 5),
            ),
            "shared_feature_recovery": metrics_sep["shared_feature_recovery"],
            "sparsity_rate": metrics_sep["sparsity"],
            "dictionary_alignment_residual": residual_sep,
            "bytes_proxy": float(
                _bytes_for_matrix(total_features, config.dim) * 2 + _bytes_for_matrix(total_features, total_features)
            ),
            "compute_proxy": float(
                _compute_proxy(
                    examples=config.train_examples,
                    dim=config.dim,
                    features=total_features,
                    iters=config.dictionary_iters * 2,
                    paired=False,
                )
            ),
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
    )

    # Shared dictionary / crosscoder.
    shared_source = train["source"].clone()
    shared_target = train["target"].clone()
    shared_decoder_source, shared_decoder_target, shared_codes, _ = _fit_shared_crosscoder(
        shared_source,
        shared_target,
        total_features,
        active=active,
        iters=config.dictionary_iters,
        lam=config.dictionary_lam,
        seed=config.seed + 211,
    )
    shared_test_codes = _infer_codes(test["source"], shared_decoder_source, active)
    shared_train_target = shared_codes @ shared_decoder_target
    shared_test_target = shared_test_codes @ shared_decoder_target
    metrics_shared = _target_projection_metrics(
        shared_test_target,
        test["target_decoder"],
        config.shared_features,
        test["shared_codes"],
        active=active,
    )
    _, residual_shared = _align_rows(shared_decoder_source, shared_decoder_target)
    rows.append(
        {
            "method": "shared_dictionary_crosscoder",
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "sparsity": float(config.sparsity),
            "noise": float(config.noise),
            "train_mse": float(F.mse_loss(shared_train_target, train["target"]).item()),
            "test_mse": float(F.mse_loss(shared_test_target, test["target"]).item()),
            "train_accuracy": _probe_accuracy(
                train["target"],
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
                shared_train_target,
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
            ),
            "test_accuracy": _probe_accuracy(
                train["target"],
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
                shared_test_target,
                _binary_labels(test["shared_codes"], seed=config.seed + 5),
            ),
            "shared_feature_recovery": metrics_shared["shared_feature_recovery"],
            "sparsity_rate": metrics_shared["sparsity"],
            "dictionary_alignment_residual": residual_shared,
            "bytes_proxy": float(
                _bytes_for_matrix(total_features, config.dim) * 2 + _bytes_for_matrix(total_features, total_features)
            ),
            "compute_proxy": float(
                _compute_proxy(
                    examples=config.train_examples,
                    dim=config.dim,
                    features=total_features,
                    iters=config.dictionary_iters,
                    paired=True,
                )
            ),
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
    )

    # Symmetry-aware shared dictionary.
    symmetry_decoder_source, symmetry_decoder_target, symmetry_codes, target_rotation = _fit_shared_crosscoder(
        train["source"],
        train["target"],
        total_features,
        active=active,
        iters=config.dictionary_iters,
        lam=config.dictionary_lam,
        seed=config.seed + 307,
        symmetry_aware=True,
    )
    symmetry_test_codes = _infer_codes(test["source"], symmetry_decoder_source, active)
    symmetry_train_target_aligned = symmetry_codes @ symmetry_decoder_target
    symmetry_test_target_aligned = symmetry_test_codes @ symmetry_decoder_target
    symmetry_test_target = symmetry_test_target_aligned @ target_rotation.T
    symmetry_train_target = symmetry_train_target_aligned @ target_rotation.T
    metrics_symmetry = _target_projection_metrics(
        symmetry_test_target,
        test["target_decoder"],
        config.shared_features,
        test["shared_codes"],
        active=active,
    )
    _, residual_symmetry = _align_rows(symmetry_decoder_source, symmetry_decoder_target)
    rows.append(
        {
            "method": "symmetry_aware_shared_dictionary",
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "sparsity": float(config.sparsity),
            "noise": float(config.noise),
            "train_mse": float(F.mse_loss(symmetry_train_target, train["target"]).item()),
            "test_mse": float(F.mse_loss(symmetry_test_target, test["target"]).item()),
            "train_accuracy": _probe_accuracy(
                train["target"],
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
                symmetry_train_target,
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
            ),
            "test_accuracy": _probe_accuracy(
                train["target"],
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
                symmetry_test_target,
                _binary_labels(test["shared_codes"], seed=config.seed + 5),
            ),
            "shared_feature_recovery": metrics_symmetry["shared_feature_recovery"],
            "sparsity_rate": metrics_symmetry["sparsity"],
            "dictionary_alignment_residual": residual_symmetry,
            "bytes_proxy": float(
                _bytes_for_matrix(total_features, config.dim) * 2 + _bytes_for_matrix(total_features, total_features)
            ),
            "compute_proxy": float(
                _compute_proxy(
                    examples=config.train_examples,
                    dim=config.dim,
                    features=total_features,
                    iters=config.dictionary_iters + 2,
                    paired=True,
                )
            ),
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
    )

    # Oracle upper bound.
    oracle_target_train = train["target"]
    oracle_target_test = test["target"]
    metrics_oracle = _target_projection_metrics(
        oracle_target_test,
        test["target_decoder"],
        config.shared_features,
        test["shared_codes"],
        active=active,
    )
    rows.append(
        {
            "method": "oracle_upper_bound",
            "seed": int(config.seed),
            "dim": int(config.dim),
            "train_examples": int(config.train_examples),
            "test_examples": int(config.test_examples),
            "shared_features": int(config.shared_features),
            "source_private_features": int(config.source_private_features),
            "target_private_features": int(config.target_private_features),
            "sparsity": float(config.sparsity),
            "noise": float(config.noise),
            "train_mse": float(F.mse_loss(oracle_target_train, train["target"]).item()),
            "test_mse": float(F.mse_loss(oracle_target_test, test["target"]).item()),
            "train_accuracy": _probe_accuracy(
                train["target"],
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
                oracle_target_train,
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
            ),
            "test_accuracy": _probe_accuracy(
                train["target"],
                _binary_labels(train["shared_codes"], seed=config.seed + 5),
                oracle_target_test,
                _binary_labels(test["shared_codes"], seed=config.seed + 5),
            ),
            "shared_feature_recovery": metrics_oracle["shared_feature_recovery"],
            "sparsity_rate": metrics_oracle["sparsity"],
            "dictionary_alignment_residual": 0.0,
            "bytes_proxy": 0.0,
            "compute_proxy": 0.0,
            "accuracy_delta_vs_raw": 0.0,
            "mse_delta_vs_raw": 0.0,
            "help_vs_raw": 0.0,
            "harm_vs_raw": 0.0,
        }
    )

    raw_accuracy = rows[0]["test_accuracy"]
    raw_mse = rows[0]["test_mse"]
    for row in rows[1:]:
        row["accuracy_delta_vs_raw"] = float(row["test_accuracy"] - raw_accuracy)
        row["mse_delta_vs_raw"] = float(row["test_mse"] - raw_mse)
        row["help_vs_raw"] = float(max(0.0, row["accuracy_delta_vs_raw"]))
        row["harm_vs_raw"] = float(max(0.0, -row["accuracy_delta_vs_raw"]))
    return rows


def _fit_shared_crosscoder(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    total_features: int,
    *,
    active: int,
    iters: int,
    lam: float,
    seed: int,
    symmetry_aware: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_rotation = torch.eye(target_train.shape[1], dtype=target_train.dtype, device=target_train.device)
    if symmetry_aware:
        source_norm = source_train - source_train.mean(dim=0, keepdim=True)
        target_norm = target_train - target_train.mean(dim=0, keepdim=True)
        align_weight, _ = _fit_ridge(source_norm, target_norm, lam)
        u, _, vt = torch.linalg.svd(align_weight, full_matrices=False)
        target_rotation = vt.T @ u.T
        target_train = target_train @ target_rotation

    gen = _make_generator(seed)
    init = _orthogonal_matrix(source_train.shape[1], gen)[:total_features].clone()
    source_decoder = _normalize_dictionary_rows(init)
    target_decoder = _normalize_dictionary_rows(init.clone())
    codes = torch.zeros(source_train.shape[0], total_features, dtype=source_train.dtype)
    eye = torch.eye(total_features, dtype=source_train.dtype)

    for _ in range(max(1, int(iters))):
        codes = _infer_joint_codes(source_train, target_train, source_decoder, target_decoder, active=active)
        source_decoder = torch.linalg.solve(codes.T @ codes + lam * eye, codes.T @ source_train)
        target_decoder = torch.linalg.solve(codes.T @ codes + lam * eye, codes.T @ target_train)
        if symmetry_aware:
            target_decoder, _ = _align_rows(source_decoder, target_decoder)
        source_decoder = _normalize_dictionary_rows(source_decoder)
        target_decoder = _normalize_dictionary_rows(target_decoder)
    codes = _infer_joint_codes(source_train, target_train, source_decoder, target_decoder, active=active)
    return source_decoder, target_decoder, codes, target_rotation


def _infer_joint_codes(
    source: torch.Tensor,
    target: torch.Tensor,
    source_decoder: torch.Tensor,
    target_decoder: torch.Tensor,
    *,
    active: int,
) -> torch.Tensor:
    design = torch.cat([source_decoder.T, target_decoder.T], dim=0)
    combined = torch.cat([source, target], dim=1)
    solution = torch.linalg.lstsq(design, combined.T).solution.T
    return _soft_threshold_rows(solution, active)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"


def write_markdown_summary(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    lines = [
        "# Toy Shared Feature Dictionary Bridge",
        "",
        f"- seed: `{rows[0]['seed'] if rows else 0}`",
        f"- dim: `{rows[0]['dim'] if rows else 0}`",
        f"- shared features: `{rows[0]['shared_features'] if rows else 0}`",
        f"- source private features: `{rows[0]['source_private_features'] if rows else 0}`",
        f"- target private features: `{rows[0]['target_private_features'] if rows else 0}`",
        "",
        "| Method | Test Acc | Test MSE | Shared recovery | Sparsity | Align residual | Bytes proxy | Compute proxy | Acc delta | MSE delta | Help vs raw | Harm vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {test_accuracy} | {test_mse} | {shared_feature_recovery} | {sparsity_rate} | {dictionary_alignment_residual} | {bytes_proxy} | {compute_proxy} | {accuracy_delta_vs_raw} | {mse_delta_vs_raw} | {help_vs_raw} | {harm_vs_raw} |".format(
                method=row["method"],
                test_accuracy=_fmt(row["test_accuracy"]),
                test_mse=_fmt(row["test_mse"]),
                shared_feature_recovery=_fmt(row["shared_feature_recovery"]),
                sparsity_rate=_fmt(row["sparsity_rate"]),
                dictionary_alignment_residual=_fmt(row["dictionary_alignment_residual"]),
                bytes_proxy=_fmt(row["bytes_proxy"]),
                compute_proxy=_fmt(row["compute_proxy"]),
                accuracy_delta_vs_raw=_fmt(row["accuracy_delta_vs_raw"]),
                mse_delta_vs_raw=_fmt(row["mse_delta_vs_raw"]),
                help_vs_raw=_fmt(row["help_vs_raw"]),
                harm_vs_raw=_fmt(row["harm_vs_raw"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy shared feature dictionary bridge ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=96)
    parser.add_argument("--test-examples", type=int, default=96)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--shared-features", type=int, default=6)
    parser.add_argument("--source-private-features", type=int, default=4)
    parser.add_argument("--target-private-features", type=int, default=4)
    parser.add_argument("--sparsity", type=float, default=0.25)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--dictionary-iters", type=int, default=8)
    parser.add_argument("--dictionary-lam", type=float, default=1e-3)
    parser.add_argument("--bridge-lam", type=float, default=1e-2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = ToySharedFeatureDictionaryConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        shared_features=args.shared_features,
        source_private_features=args.source_private_features,
        target_private_features=args.target_private_features,
        sparsity=args.sparsity,
        noise=args.noise,
        dictionary_iters=args.dictionary_iters,
        dictionary_lam=args.dictionary_lam,
        bridge_lam=args.bridge_lam,
    )
    rows = run_experiment(config)
    payload = {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
    }

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    markdown_path = pathlib.Path(args.output_md) if args.output_md else output.with_suffix(".md")
    write_markdown_summary(rows, markdown_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
