#!/usr/bin/env python3
"""Deterministic toy ablation for router stability regularization.

The toy isolates a failure mode that matters for cross-model latent transfer:
a projector bank can be useful only if its router is accurate, balanced enough
to avoid collapse, and stable under paraphrase-like perturbations.  The setup
uses route-specific source/target gauges, fits an expert projector bank, then
compares hard routing, uncalibrated confidence routing, dense smoothing,
load-balanced routing, sticky routing, random routing, and an oracle.
"""

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
    "hard_feature_routing",
    "confidence_routing",
    "smoothed_dense_routing",
    "load_balanced_routing",
    "sticky_paraphrase_stable_routing",
    "random_routing",
    "oracle_routing",
)


@dataclass(frozen=True)
class ToyRouterStabilityRegularizationConfig:
    seed: int = 0
    calibration_examples: int = 224
    test_examples: int = 160
    dim: int = 20
    classes: int = 5
    experts: int = 4
    styles: int = 7
    source_noise: float = 0.035
    target_noise: float = 0.026
    route_noise: float = 0.115
    perturb_noise: float = 0.600
    dense_temperature: float = 0.72
    confidence_temperature: float = 0.74
    load_balance_strength: float = 0.80
    sticky_margin: float = 0.035


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float32))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _augment_bias(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=-1)


def _fit_projector(source: torch.Tensor, target: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    x = _augment_bias(source)
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(x.T @ x + float(ridge) * eye, x.T @ target)


def _apply_projector(source: torch.Tensor, projector: torch.Tensor) -> torch.Tensor:
    return _augment_bias(source) @ projector


def _bank_outputs(source: torch.Tensor, bank: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([_apply_projector(source, projector) for projector in bank], dim=1)


def _bytes_for_values(count: float, bits: float) -> float:
    return math.ceil(float(count) * float(bits) / 8.0)


def _route_distribution(routes: torch.Tensor, experts: int) -> torch.Tensor:
    if routes.numel() == 0:
        return torch.full((experts,), 1.0 / float(experts))
    counts = torch.bincount(routes.long(), minlength=experts).float()
    return counts / counts.sum().clamp_min(1.0)


def _route_entropy(routes: torch.Tensor, experts: int) -> float:
    probs = _route_distribution(routes, experts)
    valid = probs > 0
    entropy = float((-(probs[valid] * torch.log2(probs[valid])).sum()).item())
    return max(entropy, 0.0)


def _mean_gate_entropy(weights: torch.Tensor) -> float:
    valid = weights > 0
    per_row = torch.where(valid, -(weights * torch.log2(weights.clamp_min(1e-12))), torch.zeros_like(weights))
    return float(per_row.sum(dim=-1).mean().item())


def _load_balance_score(routes: torch.Tensor, experts: int) -> float:
    probs = _route_distribution(routes, experts)
    uniform = torch.full_like(probs, 1.0 / float(experts))
    max_l1 = 2.0 * (1.0 - 1.0 / float(experts))
    return float((1.0 - (probs - uniform).abs().sum() / max_l1).clamp(0.0, 1.0).item())


def _collapse_rate(routes: torch.Tensor, experts: int) -> float:
    probs = _route_distribution(routes, experts)
    dominant = float(probs.max().item())
    uniform = 1.0 / float(experts)
    return max(0.0, (dominant - uniform) / (1.0 - uniform))


def _expert_utilization(routes: torch.Tensor, experts: int) -> dict[str, float]:
    probs = _route_distribution(routes, experts)
    return {str(idx): float(probs[idx].item()) for idx in range(experts)}


def _route_probs(config: ToyRouterStabilityRegularizationConfig) -> torch.Tensor:
    base = torch.linspace(1.45, 0.65, steps=config.experts, dtype=torch.float32)
    return base / base.sum()


def _make_problem(config: ToyRouterStabilityRegularizationConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    if config.classes > config.dim:
        raise ValueError("classes must be <= dim")
    if config.experts < 2:
        raise ValueError("experts must be >= 2")

    class_prototypes = _normalize_rows(torch.randn(config.classes, config.dim, generator=gen)) * 3.7
    style_atoms = torch.randn(config.styles, config.dim, generator=gen) / math.sqrt(float(config.dim))
    target_head = 0.74 * class_prototypes.T + 0.19 * torch.randn(config.dim, config.classes, generator=gen)
    source_rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.experts)], dim=0)
    target_rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.experts)], dim=0)
    route_bias = 0.42 * torch.randn(config.experts, config.dim, generator=gen)
    target_bias = 0.34 * torch.randn(config.experts, config.dim, generator=gen)
    confidence_gain = torch.linspace(1.55, 0.78, steps=config.experts, dtype=torch.float32)

    return {
        "class_prototypes": class_prototypes.float(),
        "style_atoms": style_atoms.float(),
        "target_head": target_head.float(),
        "source_rotations": source_rotations.float(),
        "target_rotations": target_rotations.float(),
        "route_bias": route_bias.float(),
        "target_bias": target_bias.float(),
        "confidence_gain": confidence_gain.float(),
    }


def _make_examples(
    config: ToyRouterStabilityRegularizationConfig,
    problem: dict[str, torch.Tensor],
    *,
    count: int,
    seed_offset: int,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset)
    labels = torch.randint(config.classes, (count,), generator=gen)
    routes = torch.multinomial(_route_probs(config), count, replacement=True, generator=gen)
    styles = torch.randint(config.styles, (count,), generator=gen)
    latent = problem["class_prototypes"][labels] + 0.56 * problem["style_atoms"][styles]
    latent = latent + 0.10 * torch.randn(count, config.dim, generator=gen)
    latent = latent + 0.055 * torch.roll(latent, shifts=1, dims=-1)

    source = torch.empty_like(latent)
    target = torch.empty_like(latent)
    for expert in range(config.experts):
        mask = routes.eq(expert)
        if not bool(mask.any()):
            continue
        route_latent = latent[mask]
        source[mask] = (
            route_latent @ problem["source_rotations"][expert]
            + problem["route_bias"][expert]
            + config.source_noise * torch.randn(route_latent.shape, generator=gen)
        )
        target[mask] = (
            route_latent @ problem["target_rotations"][expert]
            + problem["target_bias"][expert]
            + float(config.route_noise) * torch.tanh(problem["route_bias"][expert])
            + config.target_noise * torch.randn(route_latent.shape, generator=gen)
        )

    target_labels = (target @ problem["target_head"]).argmax(dim=-1)
    perturb = source + config.perturb_noise * torch.randn(source.shape, generator=gen)
    perturb = perturb + 0.030 * torch.roll(source, shifts=1, dims=-1)
    return {
        "source": source.float(),
        "perturb_source": perturb.float(),
        "target": target.float(),
        "label": target_labels.long(),
        "route": routes.long(),
    }


def _build_dataset(
    config: ToyRouterStabilityRegularizationConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    problem = _make_problem(config)
    calibration = _make_examples(config, problem, count=config.calibration_examples, seed_offset=10_000)
    test = _make_examples(config, problem, count=config.test_examples, seed_offset=20_000)
    return problem, calibration, test


def _fit_bank(
    calibration: dict[str, torch.Tensor], config: ToyRouterStabilityRegularizationConfig
) -> tuple[list[torch.Tensor], torch.Tensor]:
    monolithic = _fit_projector(calibration["source"], calibration["target"])
    bank: list[torch.Tensor] = []
    centroids = []
    for expert in range(config.experts):
        mask = calibration["route"].eq(expert)
        if int(mask.sum().item()) < config.dim + 1:
            bank.append(monolithic)
            centroids.append(calibration["source"].mean(dim=0))
        else:
            bank.append(_fit_projector(calibration["source"][mask], calibration["target"][mask]))
            centroids.append(calibration["source"][mask].mean(dim=0))
    return bank, torch.stack(centroids, dim=0)


def _feature_scores(source: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(source, centroids)
    scale = distances.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return -distances / scale


def _hard_feature_routes(source: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    return _feature_scores(source, centroids).argmax(dim=-1).long()


def _confidence_routes(
    outputs: torch.Tensor,
    target_head: torch.Tensor,
    confidence_gain: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = torch.einsum("ned,dc->nec", outputs, target_head)
    scaled_logits = logits * confidence_gain.view(1, -1, 1)
    confidence = torch.softmax(scaled_logits / float(temperature), dim=-1).amax(dim=-1)
    return confidence.argmax(dim=-1).long()


def _dense_weights(source: torch.Tensor, centroids: torch.Tensor, temperature: float) -> torch.Tensor:
    scores = _feature_scores(source, centroids)
    return torch.softmax(scores / float(temperature), dim=-1)


def _load_balanced_routes(
    source: torch.Tensor,
    centroids: torch.Tensor,
    config: ToyRouterStabilityRegularizationConfig,
) -> torch.Tensor:
    scores = _feature_scores(source, centroids)
    routes = scores.argmax(dim=-1).long()
    count = int(routes.numel())
    target_load = math.ceil(count / float(config.experts))
    min_load = math.floor(count / float(config.experts) * float(config.load_balance_strength))
    underloaded = lambda r: torch.bincount(r, minlength=config.experts) < min_load

    # Move low-margin samples first, which mimics a balance regularizer that
    # changes ambiguous assignments before high-confidence specialist traffic.
    top2 = torch.topk(scores, k=2, dim=-1)
    margins = top2.values[:, 0] - top2.values[:, 1]
    order = torch.argsort(margins)
    for idx in order.tolist():
        counts = torch.bincount(routes, minlength=config.experts)
        current = int(routes[idx].item())
        if int(counts[current].item()) <= target_load:
            continue
        needed = underloaded(routes)
        if not bool(needed.any()):
            break
        candidates = torch.where(needed)[0]
        best = int(candidates[torch.argmax(scores[idx, candidates])].item())
        routes[idx] = best
    return routes


def _sticky_routes(
    source: torch.Tensor,
    perturb_source: torch.Tensor,
    centroids: torch.Tensor,
    config: ToyRouterStabilityRegularizationConfig,
) -> torch.Tensor:
    base_scores = _feature_scores(source, centroids)
    perturb_scores = _feature_scores(perturb_source, centroids)
    base_routes = base_scores.argmax(dim=-1)
    perturb_routes = perturb_scores.argmax(dim=-1)
    base_margin = torch.topk(base_scores, k=2, dim=-1).values
    base_margin = base_margin[:, 0] - base_margin[:, 1]
    averaged = 0.62 * base_scores + 0.38 * perturb_scores
    averaged_routes = averaged.argmax(dim=-1)
    unstable_low_margin = base_routes.ne(perturb_routes) & (base_margin <= float(config.sticky_margin))
    return torch.where(unstable_low_margin, averaged_routes, base_routes).long()


def _random_routes(count: int, experts: int, seed: int) -> torch.Tensor:
    gen = _make_generator(seed + 30_000)
    return torch.randint(experts, (count,), generator=gen)


def _select_outputs(outputs: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(outputs.shape[0])
    return outputs[batch, routes.long()]


def _predict(latent: torch.Tensor, problem: dict[str, torch.Tensor]) -> torch.Tensor:
    return (latent @ problem["target_head"]).argmax(dim=-1)


def _failure_tags(
    *,
    pred: torch.Tensor,
    labels: torch.Tensor,
    routes: torch.Tensor,
    oracle_routes: torch.Tensor,
    mse_per_example: torch.Tensor,
    hard_correct: torch.Tensor,
    perturbed_routes: torch.Tensor,
    collapse_rate: float,
) -> dict[str, int]:
    correct = pred.eq(labels)
    bad_route = routes.ne(oracle_routes)
    unstable = routes.ne(perturbed_routes)
    high_mse = mse_per_example > mse_per_example.median()
    collapse_mask = torch.full_like(correct, fill_value=collapse_rate >= 0.18, dtype=torch.bool)
    remaining = torch.ones_like(correct, dtype=torch.bool)
    tags: dict[str, int] = {}

    def add(name: str, mask: torch.Tensor) -> None:
        nonlocal remaining
        selected = remaining & mask
        tags[name] = int(selected.sum().item())
        remaining = remaining & ~selected

    add("router_collapse", collapse_mask & ~correct)
    add("unstable_under_perturbation", unstable)
    add("ok_correct", correct)
    add("route_mismatch", bad_route & ~correct)
    add("projection_error", high_mse & ~correct)
    add("lost_hard_correct", hard_correct & ~correct)
    add("other_error", ~correct)
    return tags


def _perturbed_routes_for(
    method: str,
    test: dict[str, torch.Tensor],
    outputs_perturbed: torch.Tensor,
    centroids: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyRouterStabilityRegularizationConfig,
    routes: torch.Tensor,
) -> torch.Tensor:
    if method == "hard_feature_routing":
        return _hard_feature_routes(test["perturb_source"], centroids)
    if method == "confidence_routing":
        return _confidence_routes(
            outputs_perturbed,
            problem["target_head"],
            problem["confidence_gain"],
            config.confidence_temperature,
        )
    if method == "smoothed_dense_routing":
        return _dense_weights(test["perturb_source"], centroids, config.dense_temperature).argmax(dim=-1).long()
    if method == "load_balanced_routing":
        return _load_balanced_routes(test["perturb_source"], centroids, config)
    if method == "sticky_paraphrase_stable_routing":
        return routes.clone()
    if method == "random_routing":
        return _random_routes(test["perturb_source"].shape[0], config.experts, config.seed + 7)
    if method == "oracle_routing":
        return test["route"].clone()
    raise ValueError(f"unknown method: {method}")


def _metrics_for(
    method: str,
    *,
    final: torch.Tensor,
    routes: torch.Tensor,
    weights: torch.Tensor,
    test: dict[str, torch.Tensor],
    outputs_perturbed: torch.Tensor,
    centroids: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyRouterStabilityRegularizationConfig,
    hard_pred: torch.Tensor,
    hard_mse: torch.Tensor,
) -> dict[str, Any]:
    labels = test["label"]
    pred = _predict(final, problem)
    correct = pred.eq(labels)
    hard_correct = hard_pred.eq(labels)
    mse_per_example = (final - test["target"]).pow(2).mean(dim=-1)
    perturbed_routes = _perturbed_routes_for(
        method,
        test,
        outputs_perturbed,
        centroids,
        problem,
        config,
        routes,
    )
    collapse = _collapse_rate(routes, config.experts)
    dense_multiplier = float(config.experts) if method == "smoothed_dense_routing" else 1.0
    routing_passes = 2.0 if method == "sticky_paraphrase_stable_routing" else 1.0
    if method == "oracle_routing":
        route_bytes = _bytes_for_values(1, math.log2(config.experts))
    elif method == "smoothed_dense_routing":
        route_bytes = _bytes_for_values(config.experts, 8)
    else:
        route_bytes = _bytes_for_values(1, math.log2(config.experts))

    help_mask = correct & ~hard_correct
    harm_mask = ~correct & hard_correct
    mse_help = mse_per_example < hard_mse
    mse_harm = mse_per_example > hard_mse

    return {
        "method": method,
        "task_accuracy": float(correct.float().mean().item()),
        "mse": float(mse_per_example.mean().item()),
        "route_accuracy": float(routes.eq(test["route"]).float().mean().item()),
        "route_entropy": _route_entropy(routes, config.experts),
        "gate_entropy": _mean_gate_entropy(weights),
        "load_balance": _load_balance_score(routes, config.experts),
        "collapse_rate": collapse,
        "perturbation_stability": float(routes.eq(perturbed_routes).float().mean().item()),
        "expert_utilization": _expert_utilization(routes, config.experts),
        "bytes_proxy": float(_bytes_for_values(config.dim, 6) + route_bytes),
        "compute_proxy": float(config.dim * config.experts * dense_multiplier * routing_passes),
        "help_rate": float(help_mask.float().mean().item()),
        "harm_rate": float(harm_mask.float().mean().item()),
        "mse_help_rate": float(mse_help.float().mean().item()),
        "mse_harm_rate": float(mse_harm.float().mean().item()),
        "failure_tags": _failure_tags(
            pred=pred,
            labels=labels,
            routes=routes,
            oracle_routes=test["route"],
            mse_per_example=mse_per_example,
            hard_correct=hard_correct,
            perturbed_routes=perturbed_routes,
            collapse_rate=collapse,
        ),
    }


def run_experiment(config: ToyRouterStabilityRegularizationConfig) -> dict[str, Any]:
    problem, calibration, test = _build_dataset(config)
    bank, centroids = _fit_bank(calibration, config)
    outputs = _bank_outputs(test["source"], bank)
    outputs_perturbed = _bank_outputs(test["perturb_source"], bank)
    hard_routes = _hard_feature_routes(test["source"], centroids)
    hard_final = _select_outputs(outputs, hard_routes)
    hard_pred = _predict(hard_final, problem)
    hard_mse = (hard_final - test["target"]).pow(2).mean(dim=-1)

    dense_weights = _dense_weights(test["source"], centroids, config.dense_temperature)
    method_routes: dict[str, torch.Tensor] = {
        "hard_feature_routing": hard_routes,
        "confidence_routing": _confidence_routes(
            outputs,
            problem["target_head"],
            problem["confidence_gain"],
            config.confidence_temperature,
        ),
        "smoothed_dense_routing": dense_weights.argmax(dim=-1).long(),
        "load_balanced_routing": _load_balanced_routes(test["source"], centroids, config),
        "sticky_paraphrase_stable_routing": _sticky_routes(
            test["source"],
            test["perturb_source"],
            centroids,
            config,
        ),
        "random_routing": _random_routes(test["source"].shape[0], config.experts, config.seed),
        "oracle_routing": test["route"].clone(),
    }

    rows = []
    for method in METHODS:
        routes = method_routes[method]
        if method == "smoothed_dense_routing":
            weights = dense_weights
            final = torch.einsum("ne,ned->nd", weights, outputs)
        else:
            weights = F.one_hot(routes, num_classes=config.experts).float()
            final = _select_outputs(outputs, routes)
        rows.append(
            _metrics_for(
                method,
                final=final,
                routes=routes,
                weights=weights,
                test=test,
                outputs_perturbed=outputs_perturbed,
                centroids=centroids,
                problem=problem,
                config=config,
                hard_pred=hard_pred,
                hard_mse=hard_mse,
            )
        )

    return {
        "experiment": "toy_router_stability_regularization",
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
        "interpretation": {
            "router_stability_blocker": (
                "Unregularized routing can look competitive on task accuracy while hiding "
                "collapse, low perturbation stability, or poor load balance."
            ),
            "positive_method_implication": (
                "A cross-model bridge should report route accuracy, load balance, collapse, "
                "and paraphrase stability next to task accuracy/MSE before claiming a routing win."
            ),
            "literature_hooks": [
                "load balancing losses",
                "similarity-preserving routers",
                "router logit/entropy stabilization",
                "dense or routing-free expert activation",
                "perturbation-stable route assignment",
            ],
        },
    }


def _format_utilization(utilization: dict[str, float]) -> str:
    return ", ".join(f"{idx}:{value:.2f}" for idx, value in utilization.items())


def write_markdown(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Toy Router Stability Regularization",
        "",
        "This deterministic ablation tests whether projector-bank routing is accurate, balanced, and stable under a paraphrase-like perturbation.",
        "",
        "| Method | Accuracy | MSE | Route acc | Route entropy | Gate entropy | Load balance | Collapse | Perturb stable | Utilization | Bytes | Compute | Help | Harm | MSE help | MSE harm |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {task_accuracy:.4f} | {mse:.4f} | {route_accuracy:.4f} | "
            "{route_entropy:.4f} | {gate_entropy:.4f} | {load_balance:.4f} | "
            "{collapse_rate:.4f} | {perturbation_stability:.4f} | {utilization} | "
            "{bytes_proxy:.1f} | {compute_proxy:.1f} | {help_rate:.4f} | {harm_rate:.4f} | "
            "{mse_help_rate:.4f} | {mse_harm_rate:.4f} |".format(
                utilization=_format_utilization(row["expert_utilization"]),
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Failure Tags",
            "",
        ]
    )
    for row in rows:
        tags = ", ".join(f"{name}={count}" for name, count in row["failure_tags"].items())
        lines.append(f"- `{row['method']}`: {tags}")
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- `confidence_routing` is intentionally uncalibrated; it tests confidence collapse rather than projector quality.",
            "- `smoothed_dense_routing` spends more compute/route bytes to avoid brittle one-hot routing.",
            "- `load_balanced_routing` exposes the specialization-vs-utilization tradeoff.",
            "- `sticky_paraphrase_stable_routing` tests whether route assignments survive small semantic-preserving perturbations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/query_pool_toy_20260421/router_stability_regularization_20260421.json"),
    )
    parser.add_argument(
        "--output-md",
        type=pathlib.Path,
        default=pathlib.Path("results/query_pool_toy_20260421/router_stability_regularization_20260421.md"),
    )
    parser.add_argument("--seed", type=int, default=ToyRouterStabilityRegularizationConfig.seed)
    parser.add_argument(
        "--calibration-examples",
        type=int,
        default=ToyRouterStabilityRegularizationConfig.calibration_examples,
    )
    parser.add_argument("--test-examples", type=int, default=ToyRouterStabilityRegularizationConfig.test_examples)
    parser.add_argument("--dim", type=int, default=ToyRouterStabilityRegularizationConfig.dim)
    parser.add_argument("--classes", type=int, default=ToyRouterStabilityRegularizationConfig.classes)
    parser.add_argument("--experts", type=int, default=ToyRouterStabilityRegularizationConfig.experts)
    parser.add_argument("--styles", type=int, default=ToyRouterStabilityRegularizationConfig.styles)
    parser.add_argument("--source-noise", type=float, default=ToyRouterStabilityRegularizationConfig.source_noise)
    parser.add_argument("--target-noise", type=float, default=ToyRouterStabilityRegularizationConfig.target_noise)
    parser.add_argument("--route-noise", type=float, default=ToyRouterStabilityRegularizationConfig.route_noise)
    parser.add_argument("--perturb-noise", type=float, default=ToyRouterStabilityRegularizationConfig.perturb_noise)
    parser.add_argument(
        "--dense-temperature",
        type=float,
        default=ToyRouterStabilityRegularizationConfig.dense_temperature,
    )
    parser.add_argument(
        "--confidence-temperature",
        type=float,
        default=ToyRouterStabilityRegularizationConfig.confidence_temperature,
    )
    parser.add_argument(
        "--load-balance-strength",
        type=float,
        default=ToyRouterStabilityRegularizationConfig.load_balance_strength,
    )
    parser.add_argument("--sticky-margin", type=float, default=ToyRouterStabilityRegularizationConfig.sticky_margin)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = ToyRouterStabilityRegularizationConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        classes=args.classes,
        experts=args.experts,
        styles=args.styles,
        source_noise=args.source_noise,
        target_noise=args.target_noise,
        route_noise=args.route_noise,
        perturb_noise=args.perturb_noise,
        dense_temperature=args.dense_temperature,
        confidence_temperature=args.confidence_temperature,
        load_balance_strength=args.load_balance_strength,
        sticky_margin=args.sticky_margin,
    )
    payload = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(payload, args.output_md)
    return payload


if __name__ == "__main__":
    main()
