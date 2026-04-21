#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


METHODS: tuple[str, ...] = (
    "raw_pairwise_bridge",
    "monolithic_bridge",
    "hub_dictionary_only",
    "hub_feature_router",
    "hub_sticky_router",
    "hub_sticky_protected_mixed_bit_frontier",
    "hub_sticky_frontier_verifier_stop",
    "random_router_control",
    "confidence_router_control",
    "oracle_router_control",
)


ROW_KEY_ORDER: tuple[str, ...] = (
    "method",
    "route_policy",
    "seed",
    "accuracy",
    "mse",
    "accuracy_delta_vs_raw_pairwise",
    "mse_delta_vs_raw_pairwise",
    "route_accuracy",
    "route_entropy",
    "route_load",
    "perturbation_stability",
    "atom_recovery",
    "selected_atom_count",
    "protected_atom_count",
    "bit_histogram",
    "route_histogram",
    "average_stop_steps",
    "over_refinement_rate",
    "stop_reasons",
    "stop_histogram",
    "bytes_proxy",
    "compute_proxy",
    "help_vs_raw_pairwise",
    "harm_vs_raw_pairwise",
)


@dataclass(frozen=True)
class ToyHubStickyFrontierStackConfig:
    seed: int = 0
    calibration_examples: int = 192
    test_examples: int = 192
    dim: int = 20
    atoms: int = 12
    families: int = 6
    classes: int = 5
    source_noise: float = 0.030
    target_noise: float = 0.026
    route_noise: float = 0.280
    perturb_noise: float = 0.550
    route_code_strength: float = 0.72
    source_style_strength: float = 0.26
    target_style_strength: float = 0.22
    family_scale_jitter: float = 0.16
    family_bias_scale: float = 0.55
    hub_snap_strength: float = 0.35
    route_temperature: float = 0.80
    confidence_temperature: float = 0.70
    sticky_margin: float = 0.035
    keep_fraction: float = 0.70
    low_bits: int = 3
    high_bits: int = 8
    protected_atoms: int = 4
    verifier_noise: float = 0.07
    verifier_harm_margin: float = 0.012
    verifier_stop_threshold: float = 0.85
    max_steps: int = 4


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float32))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _augment_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=-1)


def _fit_linear(source: torch.Tensor, target: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    x = _augment_bias(source)
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(x.T @ x + float(ridge) * eye, x.T @ target)


def _apply_linear(source: torch.Tensor, projector: torch.Tensor) -> torch.Tensor:
    return _augment_bias(source) @ projector


def _symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _bytes_for_values(count: float, bits: float) -> int:
    return int(math.ceil(float(count) * float(bits) / 8.0))


def _estimate_bytes_for_bridge(num_projectors: int, dim: int) -> float:
    return float(num_projectors * (dim + 1) * dim * 4.0)


def _estimate_hub_bytes(config: ToyHubStickyFrontierStackConfig, protected_atoms: int, route_table: int) -> float:
    hub_params = (2 * config.families + 1) * (config.dim + 1) * config.dim
    atom_bytes = config.atoms * config.dim * 4.0
    route_bytes = max(1, route_table) * 4.0
    frontier_bytes = _bytes_for_values(max(config.atoms - protected_atoms, 0) * config.dim, config.low_bits) + _bytes_for_values(
        max(protected_atoms, 0) * config.dim, config.high_bits
    )
    return float(hub_params * 4.0 + atom_bytes + route_bytes + frontier_bytes)


def _estimate_compute_proxy(dim: int, steps: float, frontier_atoms: float, route_cost: float = 1.0) -> float:
    return float(dim * dim * route_cost + dim * max(steps, 1.0) * 1.5 + frontier_atoms * dim * 0.25)


def _route_distribution(routes: torch.Tensor, families: int) -> torch.Tensor:
    if routes.numel() == 0:
        return torch.full((families,), 1.0 / float(families), dtype=torch.float32)
    counts = torch.bincount(routes.long(), minlength=families).float()
    return counts / counts.sum().clamp_min(1.0)


def _route_entropy(routes: torch.Tensor, families: int) -> float:
    probs = _route_distribution(routes, families)
    valid = probs > 0
    entropy = float((-(probs[valid] * torch.log2(probs[valid]))).sum().item())
    return max(0.0, entropy)


def _route_load(routes: torch.Tensor, families: int) -> float:
    probs = _route_distribution(routes, families)
    uniform = torch.full_like(probs, 1.0 / float(families))
    max_l1 = 2.0 * (1.0 - 1.0 / float(families))
    return float((1.0 - (probs - uniform).abs().sum() / max_l1).clamp(0.0, 1.0).item())


def _topk_mask(scores: torch.Tensor, k: int, candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    masked = scores.clone()
    if candidate_mask is not None:
        masked[~candidate_mask] = float("-inf")
    available = int(torch.isfinite(masked).sum().item())
    k = max(0, min(int(k), available))
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if k > 0:
        indices = torch.topk(masked, k=k, largest=True).indices.sort().values
        mask[indices] = True
    return mask


def _mask_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = int(right.sum().item())
    if denom == 0:
        return 1.0 if int(left.sum().item()) == 0 else 0.0
    return float((left & right).float().sum().item() / float(denom))


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.float(), stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and float(sorted_values[end].item()) == float(sorted_values[start].item()):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2 or right.numel() < 2:
        return 0.0
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float((left_centered @ right_centered / denom).clamp(-1.0, 1.0).item())


def _select_by_threshold(scores: torch.Tensor, prior_routes: torch.Tensor, threshold: float) -> torch.Tensor:
    probs = torch.softmax(scores, dim=-1)
    confidence, route = probs.max(dim=-1)
    return torch.where(confidence >= float(threshold), route, prior_routes).long()


def _sticky_routes(
    clean_scores: torch.Tensor,
    pert_scores: torch.Tensor,
    *,
    prior_routes: torch.Tensor,
    sticky_margin: float,
) -> torch.Tensor:
    clean_prob = torch.softmax(clean_scores, dim=-1)
    clean_route = clean_prob.argmax(dim=-1)
    clean_gap = clean_prob.topk(k=2, dim=-1).values[:, 0] - clean_prob.topk(k=2, dim=-1).values[:, 1]
    pert_prob = torch.softmax(pert_scores, dim=-1)
    pert_route = pert_prob.argmax(dim=-1)
    use_prior = (pert_route != clean_route) & (clean_gap < float(sticky_margin))
    return torch.where(use_prior, prior_routes, pert_route).long()


def _make_problem(config: ToyHubStickyFrontierStackConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    if config.classes > config.dim:
        raise ValueError("classes must be <= dim")
    if config.atoms < config.protected_atoms:
        raise ValueError("atoms must be >= protected_atoms")
    if config.families < 3:
        raise ValueError("families must be >= 3")

    shared_atoms = _normalize_rows(torch.randn(config.atoms, config.dim, generator=gen)) * 3.5
    route_codes = _normalize_rows(torch.randn(config.families, config.dim, generator=gen)) * float(
        config.route_code_strength
    )
    source_style = _normalize_rows(torch.randn(config.families, config.dim, generator=gen)) * float(
        config.source_style_strength
    )
    target_style = _normalize_rows(torch.randn(config.families, config.dim, generator=gen)) * float(
        config.target_style_strength
    )
    source_rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.families)], dim=0)
    target_rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.families)], dim=0)
    source_bias = float(config.family_bias_scale) * torch.randn(config.families, config.dim, generator=gen)
    target_bias = float(config.family_bias_scale) * torch.randn(config.families, config.dim, generator=gen)
    family_scales = (1.0 + float(config.family_scale_jitter) * torch.randn(config.families, config.dim, generator=gen)).clamp_min(
        0.45
    )
    target_projection = _normalize_rows(torch.randn(config.classes, config.dim, generator=gen))
    atom_utility = torch.linspace(1.35, 0.20, steps=config.atoms, dtype=torch.float32)
    atom_utility = atom_utility - 0.80 * torch.cat(
        [torch.zeros(config.atoms // 2, dtype=torch.float32), torch.ones(config.atoms - config.atoms // 2, dtype=torch.float32)]
    )
    atom_cost = 1.0 + 0.20 * torch.arange(config.atoms, dtype=torch.float32) / max(float(config.atoms - 1), 1.0)
    atom_cost = atom_cost + 0.14 * torch.rand(config.atoms, generator=gen)
    route_permutation = torch.randperm(config.families, generator=gen)

    return {
        "shared_atoms": shared_atoms.float(),
        "route_codes": route_codes.float(),
        "source_style": source_style.float(),
        "target_style": target_style.float(),
        "source_rotations": source_rotations.float(),
        "target_rotations": target_rotations.float(),
        "source_bias": source_bias.float(),
        "target_bias": target_bias.float(),
        "family_scales": family_scales.float(),
        "target_projection": target_projection.float(),
        "atom_utility": atom_utility.float(),
        "atom_cost": atom_cost.float(),
        "route_permutation": route_permutation.long(),
    }


def _family_view(
    hub: torch.Tensor,
    family: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyHubStickyFrontierStackConfig,
    *,
    generator: torch.Generator,
    noise: float,
) -> torch.Tensor:
    out = torch.empty_like(hub)
    for family_id in range(config.families):
        mask = family.eq(family_id)
        if not bool(mask.any()):
            continue
        rotated = hub[mask] @ problem["source_rotations"][family_id]
        scaled = rotated * problem["family_scales"][family_id]
        skew = torch.tanh(hub[mask] * problem["source_style"][family_id]) * 0.10
        out[mask] = (
            scaled
            + problem["source_bias"][family_id]
            + skew
            + float(noise) * torch.randn(rotated.shape, generator=generator, dtype=rotated.dtype)
        )
    return out.float()


def _target_view(
    hub: torch.Tensor,
    family: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyHubStickyFrontierStackConfig,
    *,
    generator: torch.Generator,
    noise: float,
) -> torch.Tensor:
    out = torch.empty_like(hub)
    for family_id in range(config.families):
        mask = family.eq(family_id)
        if not bool(mask.any()):
            continue
        rotated = hub[mask] @ problem["target_rotations"][family_id]
        scaled = rotated * (problem["family_scales"][family_id] * 0.96)
        skew = torch.tanh(hub[mask] * problem["target_style"][family_id]) * 0.11
        out[mask] = (
            scaled
            + problem["target_bias"][family_id]
            + skew
            + float(noise) * torch.randn(rotated.shape, generator=generator, dtype=rotated.dtype)
        )
    return out.float()


def _sample_latents(
    config: ToyHubStickyFrontierStackConfig,
    problem: dict[str, torch.Tensor],
    *,
    count: int,
    seed_offset: int,
    source_noise: float,
    target_noise: float,
    route_noise: float,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset)
    primary = torch.randint(config.atoms, (count,), generator=gen)
    secondary = torch.randint(config.atoms - 1, (count,), generator=gen)
    secondary = torch.where(secondary >= primary, secondary + 1, secondary)
    primary_weight = 1.00 + 0.22 * torch.rand(count, generator=gen)
    secondary_weight = 0.30 + 0.14 * torch.rand(count, generator=gen)
    route_family = torch.randint(config.families, (count,), generator=gen)
    source_family = problem["route_permutation"][route_family]
    flip = torch.rand(count, generator=gen) < float(route_noise)
    if bool(flip.any()):
        random_alt = torch.randint(config.families - 1, (int(flip.sum().item()),), generator=gen)
        source_alt = torch.where(random_alt >= source_family[flip], random_alt + 1, random_alt)
        source_family = source_family.clone()
        source_family[flip] = source_alt

    hub = primary_weight.view(-1, 1) * problem["shared_atoms"][primary]
    hub = hub + secondary_weight.view(-1, 1) * problem["shared_atoms"][secondary]
    route_strength = float(config.route_code_strength) * (1.0 + 0.10 * torch.rand(count, generator=gen))
    hub = hub + route_strength.view(-1, 1) * problem["route_codes"][route_family]
    hub = hub + 0.20 * problem["source_style"][source_family] + 0.18 * problem["target_style"][route_family]
    hub = hub + float(config.hub_snap_strength) * torch.tanh(hub)
    hub = hub + 0.05 * torch.randn(count, config.dim, generator=gen)

    source = _family_view(
        hub,
        source_family,
        problem,
        config,
        generator=_make_generator(config.seed + seed_offset + 10_000),
        noise=source_noise,
    )
    target = _target_view(
        hub,
        route_family,
        problem,
        config,
        generator=_make_generator(config.seed + seed_offset + 20_000),
        noise=target_noise,
    )
    target_label = (target @ problem["target_projection"].T).argmax(dim=-1)
    perturb_source = source + float(config.perturb_noise) * torch.randn(source.shape, generator=gen, dtype=source.dtype)
    perturb_source = perturb_source + 0.028 * torch.roll(source, shifts=1, dims=-1)
    return {
        "hub": hub.float(),
        "source": source.float(),
        "target": target.float(),
        "target_label": target_label.long(),
        "perturb_source": perturb_source.float(),
        "source_family": source_family.long(),
        "target_family": route_family.long(),
        "primary_atom": primary.long(),
        "secondary_atom": secondary.long(),
    }


def _build_dataset(
    config: ToyHubStickyFrontierStackConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    problem = _make_problem(config)
    calibration = _sample_latents(
        config,
        problem,
        count=config.calibration_examples,
        seed_offset=11_000,
        source_noise=config.source_noise,
        target_noise=config.target_noise,
        route_noise=0.0,
    )
    test = _sample_latents(
        config,
        problem,
        count=config.test_examples,
        seed_offset=21_000,
        source_noise=config.source_noise,
        target_noise=config.target_noise,
        route_noise=config.route_noise,
    )
    return problem, calibration, test


def _stack(field: str, examples: dict[str, torch.Tensor]) -> torch.Tensor:
    return examples[field]


def _fit_problem_components(
    config: ToyHubStickyFrontierStackConfig,
    problem: dict[str, torch.Tensor],
    calibration: dict[str, torch.Tensor],
) -> dict[str, Any]:
    source = calibration["source"]
    target = calibration["target"]
    hub = calibration["hub"]
    source_family = calibration["source_family"]
    target_family = calibration["target_family"]

    monolithic_bridge = _fit_linear(source, target)

    pairwise_bridges: dict[tuple[int, int], torch.Tensor] = {}
    for source_id in range(config.families):
        for target_id in range(config.families):
            mask = source_family.eq(source_id) & target_family.eq(target_id)
            if int(mask.sum().item()) >= config.dim + 1:
                pairwise_bridges[(source_id, target_id)] = _fit_linear(source[mask], target[mask])
            else:
                pairwise_bridges[(source_id, target_id)] = monolithic_bridge

    source_encoders = []
    target_decoders = []
    for family_id in range(config.families):
        source_mask = source_family.eq(family_id)
        target_mask = target_family.eq(family_id)
        if int(source_mask.sum().item()) >= config.dim + 1:
            source_encoders.append(_fit_linear(source[source_mask], hub[source_mask]))
        else:
            source_encoders.append(_fit_linear(source, hub))
        if int(target_mask.sum().item()) >= config.dim + 1:
            target_decoders.append(_fit_linear(hub[target_mask], target[target_mask]))
        else:
            target_decoders.append(_fit_linear(hub, target))

    conditional_prior = torch.zeros(config.families, config.families, dtype=torch.long)
    for source_id in range(config.families):
        mask = source_family.eq(source_id)
        counts = torch.bincount(target_family[mask], minlength=config.families)
        conditional_prior[source_id] = counts
    conditional_prior_route = conditional_prior.argmax(dim=-1)

    centroids = []
    for target_id in range(config.families):
        mask = target_family.eq(target_id)
        if bool(mask.any()):
            centroids.append(source[mask].mean(dim=0))
        else:
            centroids.append(source.mean(dim=0))
    route_centroids = torch.stack(centroids, dim=0)

    atom_scores = hub @ problem["shared_atoms"].T
    atom_scores = torch.softmax(atom_scores / float(config.route_temperature), dim=-1)
    atom_quant_error = (
        problem["shared_atoms"] - _symmetric_quantize(problem["shared_atoms"], config.low_bits)
    ).pow(2).mean(dim=-1).sqrt()
    labels = (target @ problem["target_projection"].T).argmax(dim=-1)
    class_anchors = []
    for class_id in range(config.classes):
        mask = labels.eq(class_id)
        class_anchors.append(target[mask].mean(dim=0) if bool(mask.any()) else target.mean(dim=0))

    return {
        "monolithic_bridge": monolithic_bridge,
        "pairwise_bridges": pairwise_bridges,
        "source_encoders": source_encoders,
        "target_decoders": target_decoders,
        "conditional_prior_route": conditional_prior_route,
        "route_centroids": route_centroids,
        "atom_scores": atom_scores,
        "atom_quant_error": atom_quant_error,
        "class_anchors": torch.stack(class_anchors, dim=0),
    }


def _route_scores(source: torch.Tensor, route_centroids: torch.Tensor, temperature: float) -> torch.Tensor:
    distances = torch.cdist(source, route_centroids)
    scale = distances.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return -distances / scale / float(temperature)


def _route_policy(
    method: str,
    *,
    clean_source: torch.Tensor,
    perturbed_source: torch.Tensor,
    source_family: torch.Tensor,
    target_family: torch.Tensor,
    problem: dict[str, torch.Tensor],
    fitted: dict[str, Any],
    config: ToyHubStickyFrontierStackConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    route_centroids = fitted["route_centroids"]
    clean_scores = _route_scores(clean_source, route_centroids, config.route_temperature)
    pert_scores = _route_scores(perturbed_source, route_centroids, config.route_temperature)
    prior_route = fitted["conditional_prior_route"][source_family]

    if method in {"raw_pairwise_bridge", "monolithic_bridge", "hub_dictionary_only"}:
        route = prior_route
        route_policy = "conditional_prior"
    elif method == "hub_feature_router":
        route = clean_scores.argmax(dim=-1)
        route_policy = "feature_router"
    elif method == "hub_sticky_router" or method in {
        "hub_sticky_protected_mixed_bit_frontier",
        "hub_sticky_frontier_verifier_stop",
    }:
        route = _sticky_routes(clean_scores, pert_scores, prior_routes=prior_route, sticky_margin=config.sticky_margin)
        route_policy = "sticky_router"
    elif method == "random_router_control":
        gen = _make_generator(config.seed + 41_000)
        route = torch.randint(config.families, prior_route.shape, generator=gen)
        route_policy = "random_router"
    elif method == "confidence_router_control":
        route = _select_by_threshold(clean_scores, prior_route, config.confidence_temperature)
        route_policy = "confidence_router"
    elif method == "oracle_router_control":
        route = target_family.clone()
        route_policy = "oracle_router"
    else:
        raise ValueError(f"unknown method: {method}")

    route_prob = torch.softmax(clean_scores, dim=-1)
    route_confidence = route_prob.max(dim=-1).values
    return route.long(), route_prob, route_confidence, route_policy


def _decode_hub(
    hub_hat: torch.Tensor,
    route: torch.Tensor,
    fitted: dict[str, Any],
    problem: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    atom_scores = torch.softmax(hub_hat @ problem["shared_atoms"].T, dim=-1)
    atom_bank = torch.stack(
        [
            _apply_linear(problem["shared_atoms"], fitted["target_decoders"][int(route_id.item())])
            for route_id in route
        ],
        dim=0,
    )
    recon = torch.einsum("na,nad->nd", atom_scores, atom_bank)
    return recon, atom_scores


def _frontier_selection(
    atom_scores: torch.Tensor,
    route_confidence: torch.Tensor,
    problem: dict[str, torch.Tensor],
    fitted: dict[str, Any],
    *,
    protected_atoms: int,
    low_bits: int,
    high_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    per_example_keep = torch.zeros_like(atom_scores, dtype=torch.bool)
    per_example_protected = torch.zeros_like(atom_scores, dtype=torch.bool)
    bit_hist = Counter()
    for idx in range(atom_scores.shape[0]):
        scores = atom_scores[idx] * (
            0.55 * problem["atom_utility"].clamp_min(0.0)
            + 0.25 * route_confidence[idx]
            + 0.20 * (fitted["atom_quant_error"] / fitted["atom_quant_error"].clamp_min(1e-8).max())
        )
        keep_mask = _topk_mask(scores, max(int(protected_atoms) * 2, int(math.ceil(problem["shared_atoms"].shape[0] * 0.70))))
        protected_mask = _topk_mask(scores, min(int(protected_atoms), int(keep_mask.sum().item())), candidate_mask=keep_mask)
        per_example_keep[idx] = keep_mask
        per_example_protected[idx] = protected_mask
        bit_hist[str(low_bits)] += int((keep_mask & ~protected_mask).sum().item())
        bit_hist[str(high_bits)] += int(protected_mask.sum().item())
    return per_example_keep, per_example_protected, torch.tensor(
        [bit_hist.get(str(low_bits), 0), bit_hist.get(str(high_bits), 0)], dtype=torch.long
    )


def _reconstruct_from_hub(
    hub_hat: torch.Tensor,
    route: torch.Tensor,
    atom_scores: torch.Tensor,
    keep_mask: torch.Tensor | None,
    protected_mask: torch.Tensor | None,
    fitted: dict[str, Any],
    problem: dict[str, torch.Tensor],
    config: ToyHubStickyFrontierStackConfig,
) -> torch.Tensor:
    recon_rows = []
    for idx in range(hub_hat.shape[0]):
        route_id = int(route[idx].item())
        atom_bank = _apply_linear(problem["shared_atoms"], fitted["target_decoders"][route_id])
        if keep_mask is None:
            keep = torch.ones(problem["shared_atoms"].shape[0], dtype=torch.bool)
        else:
            keep = keep_mask[idx].clone()
        protected = torch.zeros_like(keep) if protected_mask is None else protected_mask[idx].clone()
        recon_bank = atom_bank.clone()
        low_mask = keep & ~protected
        high_mask = keep & protected
        if low_mask.any():
            recon_bank[low_mask] = _symmetric_quantize(recon_bank[low_mask], config.low_bits)
        if high_mask.any():
            recon_bank[high_mask] = _symmetric_quantize(recon_bank[high_mask], config.high_bits)
        recon_bank[~keep] = 0.0
        row = torch.einsum("a,ad->d", atom_scores[idx] * keep.float(), recon_bank)
        row = row / (atom_scores[idx] * keep.float()).sum().clamp_min(1e-8)
        recon_rows.append(row)
    return torch.stack(recon_rows, dim=0)


def _step_trajectory(
    hub_hat: torch.Tensor,
    route: torch.Tensor,
    atom_scores: torch.Tensor,
    frontier_keep: torch.Tensor,
    frontier_protected: torch.Tensor,
    fitted: dict[str, Any],
    problem: dict[str, torch.Tensor],
    config: ToyHubStickyFrontierStackConfig,
) -> list[torch.Tensor]:
    step0 = _reconstruct_from_hub(hub_hat, route, atom_scores, None, None, fitted, problem, config)
    step1 = _reconstruct_from_hub(hub_hat, route, atom_scores, frontier_keep, frontier_protected, fitted, problem, config)
    predicted_class = (step1 @ problem["target_projection"].T).argmax(dim=-1)
    anchor = fitted["class_anchors"][predicted_class]
    residual = anchor - step1
    step2 = step1 + 0.48 * residual
    step3 = step1 + 0.92 * residual + 0.05 * torch.roll(step1, shifts=1, dims=-1)
    return [step0, step1, step2, step3]


def _verifier_scores(
    trajectory: list[torch.Tensor],
    fitted: dict[str, Any],
    problem: dict[str, torch.Tensor],
    config: ToyHubStickyFrontierStackConfig,
) -> torch.Tensor:
    scores = []
    for step in trajectory:
        logits = step @ problem["target_projection"].T
        confidence = torch.softmax(logits / float(config.confidence_temperature), dim=-1).amax(dim=-1)
        top2 = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        predicted_class = logits.argmax(dim=-1)
        anchor = fitted["class_anchors"][predicted_class]
        residual = (step - anchor).pow(2).mean(dim=-1).sqrt()
        scores.append(confidence + 0.30 * margin - 0.20 * residual)
    return torch.stack(scores, dim=1)


def _select_by_index(trajectory: list[torch.Tensor], selected_step: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack(trajectory, dim=0)
    gather_index = selected_step.view(1, -1, 1).expand(1, -1, stacked.shape[-1])
    return stacked.gather(dim=0, index=gather_index).squeeze(0)


def _stop_indices(score_steps: torch.Tensor, config: ToyHubStickyFrontierStackConfig) -> tuple[torch.Tensor, dict[str, int]]:
    confidence = score_steps[:, 0] >= float(config.verifier_stop_threshold)
    drop = score_steps[:, 1:] < (score_steps[:, :-1] - float(config.verifier_harm_margin))
    stop_before_drop = torch.zeros_like(score_steps, dtype=torch.bool)
    stop_before_drop[:, :-1] = drop
    stop = torch.where(
        confidence,
        torch.zeros_like(confidence, dtype=torch.long),
        _first_true_or_last(stop_before_drop, last_index=score_steps.shape[1] - 1),
    )
    reasons = {
        "confidence_reached": int(confidence.sum().item()),
        "verifier_harm": int((~confidence & (stop < score_steps.shape[1] - 1)).sum().item()),
        "max_steps": int((~confidence & (stop == score_steps.shape[1] - 1)).sum().item()),
    }
    return stop.long(), reasons


def _first_true_or_last(mask: torch.Tensor, *, last_index: int) -> torch.Tensor:
    examples, steps = mask.shape
    indices = torch.arange(steps, dtype=torch.long).view(1, -1).expand(examples, -1)
    sentinel = torch.full_like(indices, fill_value=int(last_index))
    return torch.where(mask, indices, sentinel).min(dim=1).values


def _metrics_for_method(
    *,
    method: str,
    route_policy: str,
    route: torch.Tensor,
    route_prob: torch.Tensor,
    route_confidence: torch.Tensor,
    route_stability: float,
    hub_hat: torch.Tensor,
    atom_scores: torch.Tensor,
    keep_mask: torch.Tensor | None,
    protected_mask: torch.Tensor | None,
    trajectory: list[torch.Tensor],
    selected_step: torch.Tensor,
    stop_reasons: dict[str, int],
    problem: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    config: ToyHubStickyFrontierStackConfig,
    baseline_accuracy: float,
    baseline_mse: float,
    bytes_proxy: float,
    compute_proxy: float,
    bit_histogram: dict[str, int],
) -> dict[str, Any]:
    final = _select_by_index(trajectory, selected_step)
    pred = final @ problem["target_projection"].T
    pred_label = pred.argmax(dim=-1)
    target_label = test["target_label"]
    correct = pred_label.eq(target_label)
    mse = (final - test["target"]).pow(2).mean(dim=-1)

    if keep_mask is None:
        keep_count = torch.full((route.shape[0],), config.atoms, dtype=torch.long)
        prot_count = torch.zeros(route.shape[0], dtype=torch.long)
        atom_recovery = 0.0
    else:
        keep_count = keep_mask.sum(dim=-1)
        prot_count = torch.zeros_like(keep_count) if protected_mask is None else protected_mask.sum(dim=-1)
        oracle_scores = atom_scores * problem["atom_utility"].clamp_min(0.0).view(1, -1)
        oracle_keep = torch.stack([_topk_mask(oracle_scores[i], int(config.protected_atoms)) for i in range(route.shape[0])], dim=0)
        atom_recovery = (
            float(sum(_mask_overlap(protected_mask[i], oracle_keep[i]) for i in range(route.shape[0])) / max(route.shape[0], 1))
            if protected_mask is not None
            else 0.0
        )

    route_counts = torch.bincount(route.long(), minlength=config.families).tolist()
    route_hist = {str(idx): int(count) for idx, count in enumerate(route_counts) if int(count) > 0}
    route_entropy = _route_entropy(route, config.families)
    route_load = _route_load(route, config.families)
    perturbation_stability = float(route_stability)
    average_stop_steps = float((selected_step.float() + 1.0).mean().item())
    step_mse = torch.stack([(step - test["target"]).pow(2).mean(dim=-1) for step in trajectory], dim=1)
    best_step = step_mse.argmin(dim=1)
    over_refinement_rate = float(
        ((selected_step > best_step) & (mse > step_mse.min(dim=1).values + 1e-8)).float().mean().item()
    )
    accuracy_value = float(correct.float().mean().item())
    mse_value = float(mse.mean().item())

    return {
        "method": method,
        "route_policy": route_policy,
        "seed": int(config.seed),
        "accuracy": accuracy_value,
        "mse": mse_value,
        "accuracy_delta_vs_raw_pairwise": float(accuracy_value - baseline_accuracy),
        "mse_delta_vs_raw_pairwise": float(mse_value - baseline_mse),
        "route_accuracy": float(route.eq(test["target_family"]).float().mean().item()),
        "route_entropy": float(route_entropy),
        "route_load": float(route_load),
        "perturbation_stability": float(perturbation_stability),
        "atom_recovery": float(atom_recovery),
        "selected_atom_count": float(keep_count.float().mean().item()),
        "protected_atom_count": float(prot_count.float().mean().item()),
        "bit_histogram": bit_histogram,
        "route_histogram": route_hist,
        "average_stop_steps": float(average_stop_steps),
        "over_refinement_rate": float(over_refinement_rate),
        "stop_reasons": stop_reasons,
        "stop_histogram": {str(step + 1): int((selected_step == step).sum().item()) for step in range(len(trajectory))},
        "bytes_proxy": float(bytes_proxy),
        "compute_proxy": float(compute_proxy),
        "help_vs_raw_pairwise": float(max(0.0, accuracy_value - baseline_accuracy)),
        "harm_vs_raw_pairwise": float(max(0.0, baseline_accuracy - accuracy_value)),
    }


def run_experiment(config: ToyHubStickyFrontierStackConfig) -> dict[str, Any]:
    problem, calibration, test = _build_dataset(config)
    fitted = _fit_problem_components(config, problem, calibration)

    source = test["source"]
    target = test["target"]
    perturbed = test["perturb_source"]
    source_family = test["source_family"]
    target_family = test["target_family"]
    hub = test["hub"]

    route_policies: dict[str, str] = {}
    clean_routes: dict[str, torch.Tensor] = {}
    pert_routes: dict[str, torch.Tensor] = {}
    route_probs: dict[str, torch.Tensor] = {}
    route_confidence: dict[str, torch.Tensor] = {}
    route_stability: dict[str, float] = {}
    source_hubs: dict[str, torch.Tensor] = {}
    atom_scores_by_method: dict[str, torch.Tensor] = {}
    keep_masks: dict[str, torch.Tensor | None] = {}
    protected_masks: dict[str, torch.Tensor | None] = {}
    trajectories: dict[str, list[torch.Tensor]] = {}
    selected_steps: dict[str, torch.Tensor] = {}
    stop_reasons_by_method: dict[str, dict[str, int]] = {}
    bit_histograms: dict[str, dict[str, int]] = {}
    bytes_proxy_by_method: dict[str, float] = {}
    compute_proxy_by_method: dict[str, float] = {}

    for method in METHODS:
        route, prob, confidence, route_policy = _route_policy(
            method,
            clean_source=source,
            perturbed_source=perturbed,
            source_family=source_family,
            target_family=target_family,
            problem=problem,
            fitted=fitted,
            config=config,
        )
        pert_route, _, _, _ = _route_policy(
            method,
            clean_source=perturbed,
            perturbed_source=source,
            source_family=source_family,
            target_family=target_family,
            problem=problem,
            fitted=fitted,
            config=config,
        )
        route_policies[method] = route_policy
        clean_routes[method] = route
        pert_routes[method] = pert_route
        route_probs[method] = prob
        route_confidence[method] = confidence
        route_stability[method] = float((route == pert_route).float().mean().item())

        source_hub = torch.stack(
            [
                _apply_linear(source[idx : idx + 1], fitted["source_encoders"][int(source_family[idx].item())]).squeeze(0)
                for idx in range(source.shape[0])
            ],
            dim=0,
        )
        source_hubs[method] = source_hub

        atom_scores = torch.softmax(source_hub @ problem["shared_atoms"].T / float(config.route_temperature), dim=-1)
        atom_scores_by_method[method] = atom_scores

        if method in {"raw_pairwise_bridge", "monolithic_bridge"}:
            keep_masks[method] = None
            protected_masks[method] = None
            if method == "raw_pairwise_bridge":
                recon = torch.stack(
                    [
                        _apply_linear(
                            source[idx : idx + 1],
                            fitted["pairwise_bridges"][(int(source_family[idx].item()), int(target_family[idx].item()))],
                        ).squeeze(0)
                        for idx in range(source.shape[0])
                    ],
                    dim=0,
                )
            else:
                recon = _apply_linear(source, fitted["monolithic_bridge"])
            trajectories[method] = [recon]
            selected_steps[method] = torch.zeros(source.shape[0], dtype=torch.long)
            stop_reasons_by_method[method] = {"direct_bridge": int(source.shape[0])}
            bit_histograms[method] = {"16": int(config.atoms * source.shape[0])}
            if method == "raw_pairwise_bridge":
                bytes_proxy_by_method[method] = _estimate_bytes_for_bridge(
                    config.families * max(config.families - 1, 1), config.dim
                )
                compute_proxy_by_method[method] = _estimate_compute_proxy(config.dim, 1.0, 0.0, route_cost=1.20)
            else:
                bytes_proxy_by_method[method] = _estimate_bytes_for_bridge(1, config.dim)
                compute_proxy_by_method[method] = _estimate_compute_proxy(config.dim, 1.0, 0.0, route_cost=1.00)
            continue

        frontier_keep = None
        frontier_protected = None
        selected_atom_count = 0
        protected_atom_count = 0
        if method in {
            "hub_sticky_protected_mixed_bit_frontier",
            "hub_sticky_frontier_verifier_stop",
        }:
            frontier_keep, frontier_protected, histogram_counts = _frontier_selection(
                atom_scores,
                route_confidence[method],
                problem,
                fitted,
                protected_atoms=config.protected_atoms,
                low_bits=config.low_bits,
                high_bits=config.high_bits,
            )
            bit_histograms[method] = {str(config.low_bits): int(histogram_counts[0].item()), str(config.high_bits): int(histogram_counts[1].item())}
            selected_atom_count = int(frontier_keep.sum(dim=-1).float().mean().item())
            protected_atom_count = int(frontier_protected.sum(dim=-1).float().mean().item())
            keep_masks[method] = frontier_keep
            protected_masks[method] = frontier_protected
        else:
            keep_masks[method] = torch.ones_like(atom_scores, dtype=torch.bool)
            protected_masks[method] = torch.zeros_like(atom_scores, dtype=torch.bool)
            bit_histograms[method] = {str(config.low_bits): int(config.atoms * source.shape[0])}

        base_recon = _reconstruct_from_hub(
            source_hub,
            route,
            atom_scores,
            keep_masks[method],
            protected_masks[method],
            fitted,
            problem,
            config,
        )

        if method == "hub_sticky_frontier_verifier_stop":
            trajectory = _step_trajectory(
                source_hub,
                route,
                atom_scores,
                frontier_keep if frontier_keep is not None else keep_masks[method],
                frontier_protected if frontier_protected is not None else protected_masks[method],
                fitted,
                problem,
                config,
            )
            score_steps = _verifier_scores(trajectory, fitted, problem, config)
            stop_index, stop_reasons = _stop_indices(score_steps, config)
            stop_reasons_by_method[method] = stop_reasons
            selected_steps[method] = stop_index
            trajectories[method] = trajectory
            bytes_proxy_by_method[method] = _estimate_hub_bytes(config, int(protected_atom_count), config.families)
            compute_proxy_by_method[method] = _estimate_compute_proxy(
                config.dim,
                float(stop_index.float().mean().item()) + 1.0,
                float(selected_atom_count),
                route_cost=1.30,
            )
        elif method == "hub_sticky_protected_mixed_bit_frontier":
            trajectories[method] = [base_recon]
            selected_steps[method] = torch.zeros(source.shape[0], dtype=torch.long)
            stop_reasons_by_method[method] = {"frontier_fixed": int(source.shape[0])}
            bytes_proxy_by_method[method] = _estimate_hub_bytes(config, int(protected_atom_count), config.families)
            compute_proxy_by_method[method] = _estimate_compute_proxy(config.dim, 1.0, float(selected_atom_count), route_cost=1.22)
        else:
            trajectories[method] = [base_recon]
            selected_steps[method] = torch.zeros(source.shape[0], dtype=torch.long)
            stop_reasons_by_method[method] = {"low_bit_only": int(source.shape[0])}
            bytes_proxy_by_method[method] = _estimate_hub_bytes(config, 0, config.families)
            compute_proxy_by_method[method] = _estimate_compute_proxy(config.dim, 1.0, 0.0, route_cost=1.08)

    baseline_accuracy = float((trajectories["raw_pairwise_bridge"][0] @ problem["target_projection"].T).argmax(dim=-1).eq(test["target_label"]).float().mean().item())
    baseline_mse = float((trajectories["raw_pairwise_bridge"][0] - target).pow(2).mean().item())

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        row = _metrics_for_method(
            method=method,
            route_policy=route_policies[method],
            route=clean_routes[method],
            route_prob=route_probs[method],
            route_confidence=route_confidence[method],
            route_stability=route_stability[method],
            hub_hat=source_hubs[method],
            atom_scores=atom_scores_by_method[method],
            keep_mask=keep_masks[method],
            protected_mask=protected_masks[method],
            trajectory=trajectories[method],
            selected_step=selected_steps[method],
            stop_reasons=stop_reasons_by_method[method],
            problem=problem,
            test=test,
            config=config,
            baseline_accuracy=baseline_accuracy,
            baseline_mse=baseline_mse,
            bytes_proxy=bytes_proxy_by_method[method],
            compute_proxy=compute_proxy_by_method[method],
            bit_histogram=bit_histograms[method],
        )
        rows.append({key: row[key] for key in ROW_KEY_ORDER})

    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
        "interpretation": (
            "This toy composes a shared hub dictionary, route selection, sticky routing, protected mixed-bit frontiering, "
            "and verifier stopping on the same synthetic family-transfer problem. The key question is whether the stack "
            "adds gains or whether later stages interfere by erasing route stability, frontier coverage, or stop fidelity."
        ),
        "sources_consulted": [
            "https://arxiv.org/abs/2502.03714",
            "https://arxiv.org/abs/2410.06981",
            "https://arxiv.org/abs/2506.14038",
            "https://arxiv.org/abs/2204.08396",
            "https://arxiv.org/abs/2001.00281",
            "https://arxiv.org/abs/2310.05175",
            "https://arxiv.org/abs/2311.14125",
            "https://arxiv.org/abs/2501.13122",
        ],
    }


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Toy Hub Sticky Frontier Stack",
        "",
        "Deterministic composition test for shared hubs, routing stability, mixed-bit frontiers, and verifier stop rules.",
        "",
        "| Method | Accuracy | MSE | Route acc | Route entropy | Route load | Perturb stability | Atom recovery | Avg stop steps | Over-refine | Bytes proxy | Compute proxy | Help vs raw pairwise | Harm vs raw pairwise |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {mse} | {route_accuracy} | {route_entropy} | {route_load} | {perturbation_stability} | {atom_recovery} | {average_stop_steps} | {over_refinement_rate} | {bytes_proxy} | {compute_proxy} | {help_vs_raw_pairwise} | {harm_vs_raw_pairwise} |".format(
                **{key: _format_value(value) for key, value in row.items()}
            )
        )
    lines.extend(["", "## Bit Histograms", ""])
    for row in rows:
        lines.append(f"- `{row['method']}`: {json.dumps(row['bit_histogram'], sort_keys=True)}")
    lines.extend(["", "## Route Histograms", ""])
    for row in rows:
        lines.append(f"- `{row['method']}`: {json.dumps(row['route_histogram'], sort_keys=True)}")
    lines.extend(["", "## Stop Reasons", ""])
    for row in rows:
        lines.append(f"- `{row['method']}`: {json.dumps(row['stop_reasons'], sort_keys=True)}")
    lines.extend(["", "## Sources Consulted", ""])
    lines.extend(f"- {source}" for source in payload["sources_consulted"])
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--seed", type=int, default=ToyHubStickyFrontierStackConfig.seed)
    parser.add_argument("--calibration-examples", type=int, default=ToyHubStickyFrontierStackConfig.calibration_examples)
    parser.add_argument("--test-examples", type=int, default=ToyHubStickyFrontierStackConfig.test_examples)
    parser.add_argument("--dim", type=int, default=ToyHubStickyFrontierStackConfig.dim)
    parser.add_argument("--atoms", type=int, default=ToyHubStickyFrontierStackConfig.atoms)
    parser.add_argument("--families", type=int, default=ToyHubStickyFrontierStackConfig.families)
    parser.add_argument("--classes", type=int, default=ToyHubStickyFrontierStackConfig.classes)
    parser.add_argument("--source-noise", type=float, default=ToyHubStickyFrontierStackConfig.source_noise)
    parser.add_argument("--target-noise", type=float, default=ToyHubStickyFrontierStackConfig.target_noise)
    parser.add_argument("--route-noise", type=float, default=ToyHubStickyFrontierStackConfig.route_noise)
    parser.add_argument("--perturb-noise", type=float, default=ToyHubStickyFrontierStackConfig.perturb_noise)
    parser.add_argument("--route-code-strength", type=float, default=ToyHubStickyFrontierStackConfig.route_code_strength)
    parser.add_argument("--source-style-strength", type=float, default=ToyHubStickyFrontierStackConfig.source_style_strength)
    parser.add_argument("--target-style-strength", type=float, default=ToyHubStickyFrontierStackConfig.target_style_strength)
    parser.add_argument("--family-scale-jitter", type=float, default=ToyHubStickyFrontierStackConfig.family_scale_jitter)
    parser.add_argument("--family-bias-scale", type=float, default=ToyHubStickyFrontierStackConfig.family_bias_scale)
    parser.add_argument("--hub-snap-strength", type=float, default=ToyHubStickyFrontierStackConfig.hub_snap_strength)
    parser.add_argument("--route-temperature", type=float, default=ToyHubStickyFrontierStackConfig.route_temperature)
    parser.add_argument("--confidence-temperature", type=float, default=ToyHubStickyFrontierStackConfig.confidence_temperature)
    parser.add_argument("--sticky-margin", type=float, default=ToyHubStickyFrontierStackConfig.sticky_margin)
    parser.add_argument("--keep-fraction", type=float, default=ToyHubStickyFrontierStackConfig.keep_fraction)
    parser.add_argument("--low-bits", type=int, default=ToyHubStickyFrontierStackConfig.low_bits)
    parser.add_argument("--high-bits", type=int, default=ToyHubStickyFrontierStackConfig.high_bits)
    parser.add_argument("--protected-atoms", type=int, default=ToyHubStickyFrontierStackConfig.protected_atoms)
    parser.add_argument("--verifier-noise", type=float, default=ToyHubStickyFrontierStackConfig.verifier_noise)
    parser.add_argument("--verifier-harm-margin", type=float, default=ToyHubStickyFrontierStackConfig.verifier_harm_margin)
    parser.add_argument("--verifier-stop-threshold", type=float, default=ToyHubStickyFrontierStackConfig.verifier_stop_threshold)
    parser.add_argument("--max-steps", type=int, default=ToyHubStickyFrontierStackConfig.max_steps)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyHubStickyFrontierStackConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        atoms=args.atoms,
        families=args.families,
        classes=args.classes,
        source_noise=args.source_noise,
        target_noise=args.target_noise,
        route_noise=args.route_noise,
        perturb_noise=args.perturb_noise,
        route_code_strength=args.route_code_strength,
        source_style_strength=args.source_style_strength,
        target_style_strength=args.target_style_strength,
        family_scale_jitter=args.family_scale_jitter,
        family_bias_scale=args.family_bias_scale,
        hub_snap_strength=args.hub_snap_strength,
        route_temperature=args.route_temperature,
        confidence_temperature=args.confidence_temperature,
        sticky_margin=args.sticky_margin,
        keep_fraction=args.keep_fraction,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
        protected_atoms=args.protected_atoms,
        verifier_noise=args.verifier_noise,
        verifier_harm_margin=args.verifier_harm_margin,
        verifier_stop_threshold=args.verifier_stop_threshold,
        max_steps=args.max_steps,
    )
    if config.max_steps != 4:
        raise ValueError("this toy expects max_steps=4 so the verifier stop trajectory is comparable")
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(payload, pathlib.Path(args.output_md))
    return payload


if __name__ == "__main__":
    main()
