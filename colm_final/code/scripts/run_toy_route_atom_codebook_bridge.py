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
    "uniform_codebook_quantization",
    "learned_shared_codebook",
    "route_conditioned_codebook",
    "protected_outlier_atoms",
    "oracle",
)


@dataclass(frozen=True)
class ToyRouteAtomCodebookBridgeConfig:
    seed: int = 0
    train_examples: int = 160
    test_examples: int = 128
    dim: int = 16
    route_families: int = 4
    atoms_per_family: int = 4
    source_private_atoms: int = 4
    target_private_atoms: int = 4
    route_coeff_scale: float = 1.7
    private_coeff_scale: float = 0.55
    outlier_scale: float = 2.5
    noise: float = 0.06
    dictionary_iters: int = 10
    codebook_temp: float = 0.35
    ridge_lam: float = 1e-3
    protected_atoms: int = 3


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


def _bytes_for_values(count: int, bits: float) -> float:
    return float(math.ceil(count * bits / 8.0))


def _entropy_from_counts(counts: torch.Tensor) -> float:
    total = counts.sum().clamp_min(1e-8)
    probs = counts / total
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
    return float(entropy.item())


def _perplexity_from_counts(counts: torch.Tensor) -> float:
    return float(math.exp(_entropy_from_counts(counts)))


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


def _symmetric_quantize_rows(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _soft_assign(x: torch.Tensor, codebook: torch.Tensor, temp: float) -> torch.Tensor:
    scale = math.sqrt(float(x.shape[-1]))
    logits = (x @ codebook.T) / (scale * max(float(temp), 1e-8))
    return torch.softmax(logits, dim=-1)


def _apply_codebook(x: torch.Tensor, codebook: torch.Tensor, temp: float) -> tuple[torch.Tensor, torch.Tensor]:
    probs = _soft_assign(x, codebook, temp)
    recon = probs @ codebook
    return recon, probs


def _fit_codebook(
    x: torch.Tensor,
    codebook_size: int,
    *,
    iters: int,
    seed: int,
    temp: float,
    init: torch.Tensor | None = None,
) -> torch.Tensor:
    gen = _make_generator(seed)
    if init is None:
        indices = torch.randperm(x.shape[0], generator=gen)[:codebook_size]
        codebook = x[indices].clone()
    else:
        codebook = init.clone()

    if codebook.shape[0] < codebook_size:
        extra = torch.randn(codebook_size - codebook.shape[0], x.shape[1], generator=gen, dtype=x.dtype)
        codebook = torch.cat([codebook, extra], dim=0)

    codebook = codebook[:codebook_size]
    codebook = codebook + 1e-6 * torch.randn(codebook.shape, generator=gen, dtype=codebook.dtype)

    for _ in range(max(1, int(iters))):
        probs = _soft_assign(x, codebook, temp)
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


def _family_centroids(x: torch.Tensor, families: torch.Tensor, family_count: int) -> torch.Tensor:
    centroids = []
    for family in range(family_count):
        mask = families == family
        if mask.any():
            centroids.append(x[mask].mean(dim=0))
        else:
            centroids.append(x.mean(dim=0))
    return torch.stack(centroids, dim=0)


def _predict_family(x: torch.Tensor, centroids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(x, centroids)
    family = distances.argmin(dim=-1)
    family_probs = torch.softmax(-distances, dim=-1)
    return family, family_probs


def _fit_ridge(source: torch.Tensor, target: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([source, torch.ones(source.shape[0], 1, dtype=source.dtype)], dim=1)
    xtx = x.T @ x + lam * torch.eye(x.shape[1], dtype=source.dtype, device=source.device)
    xty = x.T @ target
    weight = torch.linalg.solve(xtx, xty)
    return weight[:-1], weight[-1]


def _predict_ridge(source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return source @ weight + bias


def _make_true_codebook(config: ToyRouteAtomCodebookBridgeConfig, *, seed: int) -> torch.Tensor:
    gen = _make_generator(seed)
    family_centers = _orthogonal_matrix(config.dim, gen)[: config.route_families].clone()
    atoms: list[torch.Tensor] = []
    for family in range(config.route_families):
        center = family_centers[family]
        family_scale = 1.0 + 0.25 * family
        for atom in range(config.atoms_per_family):
            direction = center + 0.35 * torch.randn(config.dim, generator=gen, dtype=torch.float32)
            direction = direction / direction.norm().clamp_min(1e-8)
            radius = config.route_coeff_scale * family_scale
            if atom == 0:
                radius *= float(config.outlier_scale)
            else:
                radius *= 0.85 + 0.12 * atom
            atoms.append(direction * radius)
    return torch.stack(atoms, dim=0)


def _make_private_basis(config: ToyRouteAtomCodebookBridgeConfig, *, seed: int, count: int) -> torch.Tensor:
    gen = _make_generator(seed)
    basis = _orthogonal_matrix(config.dim, gen)[: count].clone()
    scales = torch.linspace(0.45, 0.9, count, dtype=torch.float32)
    return basis * scales.view(-1, 1)


def _sample_split(
    config: ToyRouteAtomCodebookBridgeConfig,
    *,
    seed: int,
    examples: int,
    true_codebook: torch.Tensor,
    source_private_basis: torch.Tensor,
    target_private_basis: torch.Tensor,
    classifier: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(seed)
    total_atoms = config.route_families * config.atoms_per_family
    route_family = torch.randint(config.route_families, (examples,), generator=gen)
    local_atom = torch.randint(config.atoms_per_family, (examples,), generator=gen)
    atom_id = route_family * config.atoms_per_family + local_atom
    shared_coeff = config.route_coeff_scale + 0.35 * torch.randn(examples, generator=gen)
    shared = shared_coeff.unsqueeze(-1) * true_codebook[atom_id]

    def _sample_private(basis: torch.Tensor, count: int) -> torch.Tensor:
        active = max(1, min(count, 2))
        codes = torch.zeros(examples, count, dtype=torch.float32)
        for i in range(examples):
            idx = torch.randperm(count, generator=gen)[:active]
            codes[i, idx] = config.private_coeff_scale * torch.randn(active, generator=gen)
        return codes @ basis

    source_private = _sample_private(source_private_basis, config.source_private_atoms)
    target_private = _sample_private(target_private_basis, config.target_private_atoms)
    source = shared + source_private + config.noise * torch.randn(examples, config.dim, generator=gen)
    target = shared + target_private + config.noise * torch.randn(examples, config.dim, generator=gen)
    logits = target @ classifier + 0.15 * shared_coeff.unsqueeze(-1)
    labels = logits.argmax(dim=-1)
    return {
        "source": source.float(),
        "target": target.float(),
        "route_family": route_family.long(),
        "atom_id": atom_id.long(),
        "shared_coeff": shared_coeff.float(),
        "labels": labels.long(),
    }


def _build_problem(config: ToyRouteAtomCodebookBridgeConfig) -> dict[str, torch.Tensor]:
    true_codebook = _make_true_codebook(config, seed=config.seed + 11)
    source_private_basis = _make_private_basis(config, seed=config.seed + 17, count=config.source_private_atoms)
    target_private_basis = _make_private_basis(config, seed=config.seed + 23, count=config.target_private_atoms)
    classifier = torch.randn(config.dim, 5, generator=_make_generator(config.seed + 29), dtype=torch.float32)
    return {
        "true_codebook": true_codebook,
        "source_private_basis": source_private_basis,
        "target_private_basis": target_private_basis,
        "classifier": classifier,
        "train": _sample_split(
            config,
            seed=config.seed + 101,
            examples=config.train_examples,
            true_codebook=true_codebook,
            source_private_basis=source_private_basis,
            target_private_basis=target_private_basis,
            classifier=classifier,
        ),
        "test": _sample_split(
            config,
            seed=config.seed + 203,
            examples=config.test_examples,
            true_codebook=true_codebook,
            source_private_basis=source_private_basis,
            target_private_basis=target_private_basis,
            classifier=classifier,
        ),
    }


def _code_stats(assignments: torch.Tensor, codebook_size: int) -> tuple[float, float]:
    assignments = assignments.to(dtype=torch.long)
    counts = torch.bincount(assignments.view(-1), minlength=codebook_size).float()
    entropy = _entropy_from_counts(counts)
    perplexity = _perplexity_from_counts(counts)
    return entropy, perplexity


def _predict_accuracy(predicted_target: torch.Tensor, classifier: torch.Tensor, labels: torch.Tensor) -> float:
    logits = predicted_target @ classifier
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def _bytes_proxy_raw(config: ToyRouteAtomCodebookBridgeConfig) -> float:
    return float(4 * config.dim * (config.dim + 1))


def _bytes_proxy_codebook(
    *,
    codebook_size: int,
    dim: int,
    protected_atoms: int = 0,
    families: int = 1,
) -> float:
    codebook_bytes = _bytes_for_values(codebook_size * dim, 8.0)
    protected_bytes = _bytes_for_values(protected_atoms * dim, 16.0)
    family_bytes = _bytes_for_values(families * dim, 8.0)
    return float(codebook_bytes + protected_bytes + family_bytes + 8.0)


def _compute_proxy(
    *,
    examples: int,
    dim: int,
    codebook_size: int | None,
    families: int = 1,
    protected_atoms: int = 0,
    ridge_dim: int | None = None,
) -> float:
    ridge_dim = dim if ridge_dim is None else int(ridge_dim)
    if codebook_size is None:
        return float(examples * ridge_dim * dim)
    return float(
        examples * codebook_size * dim
        + families * examples * dim
        + protected_atoms * dim * dim
    )


def _fit_and_score(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    source_test: torch.Tensor,
    target_test: torch.Tensor,
    classifier: torch.Tensor,
    *,
    lam: float,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    weight, bias = _fit_ridge(source_train, target_train, lam)
    pred = _predict_ridge(source_test, weight, bias)
    metrics = {
        "mse": float(F.mse_loss(pred, target_test).item()),
        "accuracy": _predict_accuracy(pred, classifier, (target_test @ classifier).argmax(dim=-1)),
    }
    return metrics, weight, bias


def _evaluate_method(
    config: ToyRouteAtomCodebookBridgeConfig,
    problem: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    train = problem["train"]
    test = problem["test"]
    true_codebook = problem["true_codebook"]
    classifier = problem["classifier"]
    source_train = train["source"]
    source_test = test["source"]
    target_train = train["target"]
    target_test = test["target"]

    shared_atoms = config.route_families * config.atoms_per_family
    uniform_codebook = _normalize_rows(_orthogonal_matrix(config.dim, _make_generator(config.seed + 301))[:shared_atoms])
    learned_codebook = _fit_codebook(
        source_train,
        shared_atoms,
        iters=config.dictionary_iters,
        seed=config.seed + 401,
        temp=config.codebook_temp,
    )
    family_centroids = _family_centroids(source_train, train["route_family"], config.route_families)
    route_family_pred, route_family_probs = _predict_family(source_test, family_centroids)

    route_codebooks = []
    for family in range(config.route_families):
        mask = train["route_family"] == family
        family_source = source_train[mask] if mask.any() else source_train
        route_codebooks.append(
            _fit_codebook(
                family_source,
                config.atoms_per_family,
                iters=config.dictionary_iters,
                seed=config.seed + 503 + family,
                temp=config.codebook_temp,
            )
        )
    route_codebooks_tensor = torch.stack(route_codebooks, dim=0)

    protected_order = learned_codebook.norm(dim=1).argsort(descending=True)
    protected_idx = protected_order[: max(1, min(config.protected_atoms, shared_atoms))]
    quantized_learned = learned_codebook.clone()
    bulk_idx = torch.tensor([i for i in range(shared_atoms) if i not in set(protected_idx.tolist())], dtype=torch.long)
    if bulk_idx.numel() > 0:
        quantized_learned[bulk_idx] = _symmetric_quantize_rows(quantized_learned[bulk_idx], 8)

    def _quantized_recon(x: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon, probs = _apply_codebook(x, codebook, config.codebook_temp)
        return recon, probs.argmax(dim=-1)

    if method == "raw_ridge":
        metrics, _, _ = _fit_and_score(
            source_train,
            target_train,
            source_test,
            target_test,
            classifier,
            lam=config.ridge_lam,
        )
        row = {
            "method": method,
            "accuracy": metrics["accuracy"],
            "mse": metrics["mse"],
            "atom_recovery": 0.0,
            "codebook_entropy": _entropy_from_counts(torch.bincount(train["atom_id"], minlength=shared_atoms).float()),
            "codebook_perplexity": _perplexity_from_counts(torch.bincount(train["atom_id"], minlength=shared_atoms).float()),
            "bytes_proxy": _bytes_proxy_raw(config),
            "compute_proxy": _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                codebook_size=None,
                ridge_dim=config.dim,
            ),
        }
        return row

    if method == "uniform_codebook_quantization":
        recon_train, train_atoms = _quantized_recon(source_train, uniform_codebook)
        recon_test, test_atoms = _quantized_recon(source_test, uniform_codebook)
        metrics, _, _ = _fit_and_score(
            recon_train,
            target_train,
            recon_test,
            target_test,
            classifier,
            lam=config.ridge_lam,
        )
        recovery = _match_recovery(true_codebook, uniform_codebook)
        entropy, perplexity = _code_stats(test_atoms, shared_atoms)
        row = {
            "method": method,
            "accuracy": metrics["accuracy"],
            "mse": metrics["mse"],
            "atom_recovery": recovery,
            "codebook_entropy": entropy,
            "codebook_perplexity": perplexity,
            "bytes_proxy": _bytes_proxy_codebook(codebook_size=shared_atoms, dim=config.dim),
            "compute_proxy": _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                codebook_size=shared_atoms,
            ),
        }
        return row

    if method == "learned_shared_codebook":
        recon_train, train_atoms = _quantized_recon(source_train, learned_codebook)
        recon_test, test_atoms = _quantized_recon(source_test, learned_codebook)
        metrics, _, _ = _fit_and_score(
            recon_train,
            target_train,
            recon_test,
            target_test,
            classifier,
            lam=config.ridge_lam,
        )
        recovery = _match_recovery(true_codebook, learned_codebook)
        entropy, perplexity = _code_stats(test_atoms, shared_atoms)
        row = {
            "method": method,
            "accuracy": metrics["accuracy"],
            "mse": metrics["mse"],
            "atom_recovery": recovery,
            "codebook_entropy": entropy,
            "codebook_perplexity": perplexity,
            "bytes_proxy": _bytes_proxy_codebook(codebook_size=shared_atoms, dim=config.dim),
            "compute_proxy": _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                codebook_size=shared_atoms,
            ),
        }
        return row

    if method == "route_conditioned_codebook":
        recon_train_parts = []
        train_atoms_parts = []
        for family in range(config.route_families):
            mask = train["route_family"] == family
            if not mask.any():
                continue
            recon, probs = _apply_codebook(source_train[mask], route_codebooks_tensor[family], config.codebook_temp)
            recon_train_parts.append((mask, recon))
            train_atoms_parts.append(probs.argmax(dim=-1) + family * config.atoms_per_family)

        route_recon_test = []
        route_atoms_test = []
        for family in range(config.route_families):
            mask = route_family_pred == family
            if not mask.any():
                continue
            recon, probs = _apply_codebook(source_test[mask], route_codebooks_tensor[family], config.codebook_temp)
            route_recon_test.append((mask, recon))
            route_atoms_test.append(probs.argmax(dim=-1) + family * config.atoms_per_family)

        recon_train = source_train.clone()
        recon_test = source_test.clone()
        for mask, recon in recon_train_parts:
            recon_train[mask] = recon
        for mask, recon in route_recon_test:
            recon_test[mask] = recon

        metrics, _, _ = _fit_and_score(
            recon_train,
            target_train,
            recon_test,
            target_test,
            classifier,
            lam=config.ridge_lam,
        )
        route_atoms = torch.cat(route_atoms_test, dim=0) if route_atoms_test else torch.zeros(0, dtype=torch.long)
        route_atoms = route_atoms.reshape(-1).to(dtype=torch.long)
        entropy, perplexity = _code_stats(route_atoms, shared_atoms)
        recovery = sum(
            _match_recovery(true_codebook[family * config.atoms_per_family : (family + 1) * config.atoms_per_family], route_codebooks_tensor[family])
            for family in range(config.route_families)
        ) / max(config.route_families, 1)
        row = {
            "method": method,
            "accuracy": metrics["accuracy"],
            "mse": metrics["mse"],
            "atom_recovery": recovery,
            "codebook_entropy": entropy,
            "codebook_perplexity": perplexity,
            "bytes_proxy": _bytes_proxy_codebook(
                codebook_size=shared_atoms,
                dim=config.dim,
                families=config.route_families,
            ),
            "compute_proxy": _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                codebook_size=config.atoms_per_family,
                families=config.route_families,
            ),
            "route_entropy": _entropy_from_counts(torch.bincount(route_family_pred, minlength=config.route_families).float()),
            "route_perplexity": _perplexity_from_counts(torch.bincount(route_family_pred, minlength=config.route_families).float()),
        }
        return row

    if method == "protected_outlier_atoms":
        recon_train, train_probs = _apply_codebook(source_train, quantized_learned, config.codebook_temp)
        recon_test, test_probs = _apply_codebook(source_test, quantized_learned, config.codebook_temp)
        train_atoms = train_probs.argmax(dim=-1)
        test_atoms = test_probs.argmax(dim=-1)
        protected_set = set(protected_idx.tolist())
        for row_idx, atom_idx in enumerate(test_atoms.tolist()):
            if atom_idx in protected_set:
                recon_test[row_idx] = source_test[row_idx]
        for row_idx, atom_idx in enumerate(train_atoms.tolist()):
            if atom_idx in protected_set:
                recon_train[row_idx] = source_train[row_idx]
        metrics, _, _ = _fit_and_score(
            recon_train,
            target_train,
            recon_test,
            target_test,
            classifier,
            lam=config.ridge_lam,
        )
        recovery = _match_recovery(true_codebook, quantized_learned)
        entropy, perplexity = _code_stats(test_atoms, shared_atoms)
        row = {
            "method": method,
            "accuracy": metrics["accuracy"],
            "mse": metrics["mse"],
            "atom_recovery": recovery,
            "codebook_entropy": entropy,
            "codebook_perplexity": perplexity,
            "bytes_proxy": _bytes_proxy_codebook(
                codebook_size=shared_atoms,
                dim=config.dim,
                protected_atoms=len(protected_set),
            ),
            "compute_proxy": _compute_proxy(
                examples=config.train_examples,
                dim=config.dim,
                codebook_size=shared_atoms,
                protected_atoms=len(protected_set),
            ),
        }
        return row

    if method == "oracle":
        metrics, _, _ = _fit_and_score(
            target_train,
            target_train,
            target_test,
            target_test,
            classifier,
            lam=config.ridge_lam,
        )
        row = {
            "method": method,
            "accuracy": 1.0,
            "mse": 0.0,
            "atom_recovery": 1.0,
            "codebook_entropy": _entropy_from_counts(torch.bincount(train["atom_id"], minlength=shared_atoms).float()),
            "codebook_perplexity": _perplexity_from_counts(torch.bincount(train["atom_id"], minlength=shared_atoms).float()),
            "bytes_proxy": float(_bytes_for_values(shared_atoms * config.dim, 16.0)),
            "compute_proxy": float(config.train_examples * config.dim),
        }
        return row

    raise ValueError(f"Unknown method: {method}")


def run_experiment(config: ToyRouteAtomCodebookBridgeConfig) -> list[dict[str, Any]]:
    problem = _build_problem(config)
    rows = [_evaluate_method(config, problem, method) for method in METHODS]
    raw_accuracy = rows[0]["accuracy"]
    for row in rows:
        delta = float(row["accuracy"] - raw_accuracy)
        row["help_vs_raw"] = max(0.0, delta)
        row["harm_vs_raw"] = max(0.0, -delta)
        row["accuracy_delta_vs_raw"] = delta
        row["mse_delta_vs_raw"] = float(row["mse"] - rows[0]["mse"])
    return rows


def write_markdown_summary(config: ToyRouteAtomCodebookBridgeConfig, rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    lines = [
        "# Toy Route Atom Codebook Bridge",
        "",
        f"- Seed: `{config.seed}`",
        f"- Train examples: `{config.train_examples}`",
        f"- Test examples: `{config.test_examples}`",
        f"- Route families: `{config.route_families}`",
        f"- Atoms per family: `{config.atoms_per_family}`",
        f"- Protected atoms: `{config.protected_atoms}`",
        "",
        "| Method | Accuracy | MSE | Atom recovery | Codebook entropy | Codebook perplexity | Bytes proxy | Compute proxy | Help vs raw | Harm vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {mse} | {atom_recovery} | {codebook_entropy} | {codebook_perplexity} | {bytes_proxy} | {compute_proxy} | {help_vs_raw} | {harm_vs_raw} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                mse=fmt(row["mse"]),
                atom_recovery=fmt(row["atom_recovery"]),
                codebook_entropy=fmt(row["codebook_entropy"]),
                codebook_perplexity=fmt(row["codebook_perplexity"]),
                bytes_proxy=fmt(row["bytes_proxy"]),
                compute_proxy=fmt(row["compute_proxy"]),
                help_vs_raw=fmt(row["help_vs_raw"]),
                harm_vs_raw=fmt(row["harm_vs_raw"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy route-atom codebook bridge ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=160)
    parser.add_argument("--test-examples", type=int, default=128)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--route-families", type=int, default=4)
    parser.add_argument("--atoms-per-family", type=int, default=4)
    parser.add_argument("--source-private-atoms", type=int, default=4)
    parser.add_argument("--target-private-atoms", type=int, default=4)
    parser.add_argument("--route-coeff-scale", type=float, default=1.7)
    parser.add_argument("--private-coeff-scale", type=float, default=0.55)
    parser.add_argument("--outlier-scale", type=float, default=2.5)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("--dictionary-iters", type=int, default=10)
    parser.add_argument("--codebook-temp", type=float, default=0.35)
    parser.add_argument("--ridge-lam", type=float, default=1e-3)
    parser.add_argument("--protected-atoms", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyRouteAtomCodebookBridgeConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        route_families=args.route_families,
        atoms_per_family=args.atoms_per_family,
        source_private_atoms=args.source_private_atoms,
        target_private_atoms=args.target_private_atoms,
        route_coeff_scale=args.route_coeff_scale,
        private_coeff_scale=args.private_coeff_scale,
        outlier_scale=args.outlier_scale,
        noise=args.noise,
        dictionary_iters=args.dictionary_iters,
        codebook_temp=args.codebook_temp,
        ridge_lam=args.ridge_lam,
        protected_atoms=args.protected_atoms,
    )
    rows = run_experiment(config)
    payload = {"config": asdict(config), "methods": list(METHODS), "rows": rows}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        write_markdown_summary(config, rows, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
