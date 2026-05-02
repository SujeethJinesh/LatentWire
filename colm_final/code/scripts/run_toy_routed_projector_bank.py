#!/usr/bin/env python3
"""Deterministic toy ablation for routed projector banks.

The setup mirrors a multimodal bridge: source latents arrive in several
route-specific gauges, and the target model needs a projector matched to the
route.  The toy compares a no-route baseline, a monolithic projector, oracle
routing, confidence routing, feature routing, and deterministic random routing.
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
    "no_route_baseline",
    "monolithic_projector",
    "oracle_routed_bank",
    "confidence_routed_bank",
    "feature_routed_bank",
    "random_routed_bank",
)


@dataclass(frozen=True)
class ToyRoutedProjectorBankConfig:
    seed: int = 0
    calibration_examples: int = 192
    test_examples: int = 160
    dim: int = 20
    classes: int = 5
    experts: int = 4
    styles: int = 6
    source_noise: float = 0.035
    target_noise: float = 0.025
    route_noise: float = 0.10
    perturb_noise: float = 0.025
    confidence_temperature: float = 0.85


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


def _entropy_from_routes(routes: torch.Tensor, experts: int) -> float:
    if routes.numel() == 0:
        return 0.0
    counts = torch.bincount(routes.long(), minlength=experts).float()
    probs = counts / counts.sum().clamp_min(1.0)
    valid = probs > 0
    entropy = float((-(probs[valid] * torch.log2(probs[valid])).sum()).item())
    return 0.0 if abs(entropy) < 1e-12 else max(entropy, 0.0)


def _utilization(routes: torch.Tensor, experts: int) -> dict[str, float]:
    if routes.numel() == 0:
        return {str(idx): 0.0 for idx in range(experts)}
    counts = torch.bincount(routes.long(), minlength=experts).float()
    probs = counts / counts.sum().clamp_min(1.0)
    return {str(idx): float(probs[idx].item()) for idx in range(experts)}


def _make_problem(config: ToyRoutedProjectorBankConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    if config.classes > config.dim:
        raise ValueError("classes must be <= dim")
    if config.experts < 2:
        raise ValueError("experts must be >= 2")

    class_prototypes = _normalize_rows(torch.randn(config.classes, config.dim, generator=gen)) * 3.8
    style_atoms = torch.randn(config.styles, config.dim, generator=gen) / math.sqrt(float(config.dim))
    target_head = 0.72 * class_prototypes.T + 0.20 * torch.randn(config.dim, config.classes, generator=gen)
    source_rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.experts)], dim=0)
    target_rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.experts)], dim=0)
    route_bias = 0.65 * torch.randn(config.experts, config.dim, generator=gen)
    target_bias = 0.35 * torch.randn(config.experts, config.dim, generator=gen)

    return {
        "class_prototypes": class_prototypes.float(),
        "style_atoms": style_atoms.float(),
        "target_head": target_head.float(),
        "source_rotations": source_rotations.float(),
        "target_rotations": target_rotations.float(),
        "route_bias": route_bias.float(),
        "target_bias": target_bias.float(),
    }


def _make_examples(
    config: ToyRoutedProjectorBankConfig,
    problem: dict[str, torch.Tensor],
    *,
    count: int,
    seed_offset: int,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset)
    labels = torch.randint(config.classes, (count,), generator=gen)
    routes = torch.randint(config.experts, (count,), generator=gen)
    styles = torch.randint(config.styles, (count,), generator=gen)
    latent = problem["class_prototypes"][labels] + 0.58 * problem["style_atoms"][styles]
    latent = latent + 0.09 * torch.randn(count, config.dim, generator=gen)
    latent = latent + 0.06 * torch.roll(latent, shifts=1, dims=-1)

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

    labels = (target @ problem["target_head"]).argmax(dim=-1)
    return {
        "source": source.float(),
        "target": target.float(),
        "label": labels.long(),
        "route": routes.long(),
    }


def _build_dataset(
    config: ToyRoutedProjectorBankConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    problem = _make_problem(config)
    calibration = _make_examples(config, problem, count=config.calibration_examples, seed_offset=10_000)
    test = _make_examples(config, problem, count=config.test_examples, seed_offset=20_000)
    return problem, calibration, test


def _fit_bank(
    calibration: dict[str, torch.Tensor], config: ToyRoutedProjectorBankConfig
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
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
    return monolithic, bank, torch.stack(centroids, dim=0)


def _bank_outputs(source: torch.Tensor, bank: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([_apply_projector(source, projector) for projector in bank], dim=1)


def _feature_routes(source: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(source, centroids)
    return distances.argmin(dim=-1).long()


def _confidence_routes(outputs: torch.Tensor, target_head: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = torch.einsum("ned,dc->nec", outputs, target_head)
    confidence = torch.softmax(logits / float(temperature), dim=-1).amax(dim=-1)
    return confidence.argmax(dim=-1).long()


def _random_routes(count: int, experts: int, seed: int) -> torch.Tensor:
    gen = _make_generator(seed + 30_000)
    return torch.randint(experts, (count,), generator=gen)


def _select_outputs(outputs: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(outputs.shape[0])
    return outputs[batch, routes.long()]


def _failure_tags(
    pred: torch.Tensor,
    label: torch.Tensor,
    routes: torch.Tensor,
    oracle_routes: torch.Tensor,
    mse_per_example: torch.Tensor,
    baseline_correct: torch.Tensor,
) -> dict[str, int]:
    correct = pred.eq(label)
    bad_route = routes.ne(oracle_routes)
    high_mse = mse_per_example > mse_per_example.median()
    remaining = torch.ones_like(correct, dtype=torch.bool)
    tags: dict[str, int] = {}

    def add(name: str, mask: torch.Tensor) -> None:
        nonlocal remaining
        selected = remaining & mask
        tags[name] = int(selected.sum().item())
        remaining = remaining & ~selected

    add("ok_correct", correct)
    add("route_mismatch", bad_route & ~correct)
    add("projection_error", high_mse & ~correct)
    add("lost_baseline_correct", baseline_correct & ~correct)
    add("wrong_low_mse", ~high_mse & ~correct)
    tags["other"] = int(remaining.sum().item())
    return tags


def _metrics_for(
    method: str,
    reconstruction: torch.Tensor,
    routes: torch.Tensor,
    problem: dict[str, torch.Tensor],
    test: dict[str, torch.Tensor],
    config: ToyRoutedProjectorBankConfig,
    *,
    baseline_correct: torch.Tensor,
    baseline_mse: torch.Tensor,
    bytes_proxy: float,
    compute_proxy: float,
    centroids: torch.Tensor,
    bank: Sequence[torch.Tensor],
) -> dict[str, Any]:
    logits = reconstruction @ problem["target_head"]
    pred = logits.argmax(dim=-1)
    correct = pred.eq(test["label"])
    mse_per_example = (reconstruction - test["target"]).pow(2).mean(dim=-1)
    perturb_gen = _make_generator(config.seed + 40_000)
    perturbed_source = test["source"] + config.perturb_noise * torch.randn(
        test["source"].shape, generator=perturb_gen, dtype=test["source"].dtype
    )
    perturbed_outputs = _bank_outputs(perturbed_source, bank)
    if method == "confidence_routed_bank":
        perturbed_routes = _confidence_routes(perturbed_outputs, problem["target_head"], config.confidence_temperature)
    elif method == "feature_routed_bank":
        perturbed_routes = _feature_routes(perturbed_source, centroids)
    elif method == "random_routed_bank":
        perturbed_routes = routes
    elif method == "oracle_routed_bank":
        perturbed_routes = test["route"]
    else:
        perturbed_routes = routes

    return {
        "method": method,
        "task_accuracy": float(correct.float().mean().item()),
        "mse": float(mse_per_example.mean().item()),
        "route_entropy": _entropy_from_routes(routes, config.experts),
        "expert_utilization": _utilization(routes, config.experts),
        "route_stability": float(routes.eq(perturbed_routes).float().mean().item()),
        "help_vs_no_route": float(((~baseline_correct) & correct).float().mean().item()),
        "harm_vs_no_route": float((baseline_correct & ~correct).float().mean().item()),
        "mse_help_vs_no_route": float((mse_per_example < baseline_mse - 1e-8).float().mean().item()),
        "mse_harm_vs_no_route": float((mse_per_example > baseline_mse + 1e-8).float().mean().item()),
        "route_accuracy": float(routes.eq(test["route"]).float().mean().item()),
        "bytes_proxy": float(bytes_proxy),
        "compute_proxy": float(compute_proxy),
        "failure_tags": _failure_tags(pred, test["label"], routes, test["route"], mse_per_example, baseline_correct),
    }


def run_experiment(config: ToyRoutedProjectorBankConfig) -> dict[str, Any]:
    problem, calibration, test = _build_dataset(config)
    monolithic, bank, centroids = _fit_bank(calibration, config)
    bank_outputs = _bank_outputs(test["source"], bank)
    no_route_recon = test["source"]
    baseline_logits = no_route_recon @ problem["target_head"]
    baseline_correct = baseline_logits.argmax(dim=-1).eq(test["label"])
    baseline_mse = (no_route_recon - test["target"]).pow(2).mean(dim=-1)

    single_route = torch.zeros(config.test_examples, dtype=torch.long)
    monolithic_recon = _apply_projector(test["source"], monolithic)
    oracle_routes = test["route"]
    confidence_routes = _confidence_routes(bank_outputs, problem["target_head"], config.confidence_temperature)
    feature_routes = _feature_routes(test["source"], centroids)
    random_routes = _random_routes(config.test_examples, config.experts, config.seed)

    dim = float(config.dim)
    bank_bytes = config.experts * ((config.dim + 1) * config.dim * 4.0)
    rows = [
        _metrics_for(
            "no_route_baseline",
            no_route_recon,
            single_route,
            problem,
            test,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            bytes_proxy=dim * 4.0,
            compute_proxy=dim,
            centroids=centroids,
            bank=bank,
        ),
        _metrics_for(
            "monolithic_projector",
            monolithic_recon,
            single_route,
            problem,
            test,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            bytes_proxy=(config.dim + 1) * config.dim * 4.0,
            compute_proxy=dim * dim,
            centroids=centroids,
            bank=bank,
        ),
        _metrics_for(
            "oracle_routed_bank",
            _select_outputs(bank_outputs, oracle_routes),
            oracle_routes,
            problem,
            test,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            bytes_proxy=bank_bytes,
            compute_proxy=dim * dim + config.experts,
            centroids=centroids,
            bank=bank,
        ),
        _metrics_for(
            "confidence_routed_bank",
            _select_outputs(bank_outputs, confidence_routes),
            confidence_routes,
            problem,
            test,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            bytes_proxy=bank_bytes + config.experts * config.classes * 2.0,
            compute_proxy=config.experts * dim * dim,
            centroids=centroids,
            bank=bank,
        ),
        _metrics_for(
            "feature_routed_bank",
            _select_outputs(bank_outputs, feature_routes),
            feature_routes,
            problem,
            test,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            bytes_proxy=bank_bytes + config.experts * dim * 4.0,
            compute_proxy=config.experts * dim + dim * dim,
            centroids=centroids,
            bank=bank,
        ),
        _metrics_for(
            "random_routed_bank",
            _select_outputs(bank_outputs, random_routes),
            random_routes,
            problem,
            test,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            bytes_proxy=bank_bytes,
            compute_proxy=dim * dim,
            centroids=centroids,
            bank=bank,
        ),
    ]

    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
        "interpretation": (
            "A routed projector bank is useful when source latents occupy route-specific gauges. "
            "Oracle routing upper-bounds the bank; feature routing tests cheap centroid routing; "
            "confidence routing tests target-head self-selection; random routing isolates bank capacity from route quality."
        ),
    }


def _format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = [
        "# Toy Routed Projector Bank",
        "",
        "Deterministic multimodal-inspired ablation for route-specific projector banks in cross-model latent exchange.",
        "",
        "| Method | Accuracy | MSE | Route acc | Route entropy | Route stability | Expert utilization | Bytes proxy | Compute proxy | Help | Harm | MSE help | MSE harm |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        utilization = ", ".join(f"{key}:{value:.2f}" for key, value in row["expert_utilization"].items())
        lines.append(
            "| {method} | {task_accuracy} | {mse} | {route_accuracy} | {route_entropy} | {route_stability} | "
            "`{utilization}` | {bytes_proxy} | {compute_proxy} | {help_vs_no_route} | {harm_vs_no_route} | "
            "{mse_help_vs_no_route} | {mse_harm_vs_no_route} |".format(
                utilization=utilization,
                **{key: _format_float(value) for key, value in row.items() if key != "failure_tags"},
            )
        )
    lines.extend(["", "## Failure Tags", ""])
    for row in payload["rows"]:
        tags = ", ".join(f"{key}={value}" for key, value in sorted(row["failure_tags"].items()))
        lines.append(f"- `{row['method']}`: {tags}")
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--output-md", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=ToyRoutedProjectorBankConfig.seed)
    parser.add_argument(
        "--calibration-examples", type=int, default=ToyRoutedProjectorBankConfig.calibration_examples
    )
    parser.add_argument("--test-examples", type=int, default=ToyRoutedProjectorBankConfig.test_examples)
    parser.add_argument("--dim", type=int, default=ToyRoutedProjectorBankConfig.dim)
    parser.add_argument("--classes", type=int, default=ToyRoutedProjectorBankConfig.classes)
    parser.add_argument("--experts", type=int, default=ToyRoutedProjectorBankConfig.experts)
    parser.add_argument("--styles", type=int, default=ToyRoutedProjectorBankConfig.styles)
    parser.add_argument("--source-noise", type=float, default=ToyRoutedProjectorBankConfig.source_noise)
    parser.add_argument("--target-noise", type=float, default=ToyRoutedProjectorBankConfig.target_noise)
    parser.add_argument("--route-noise", type=float, default=ToyRoutedProjectorBankConfig.route_noise)
    parser.add_argument("--perturb-noise", type=float, default=ToyRoutedProjectorBankConfig.perturb_noise)
    parser.add_argument(
        "--confidence-temperature", type=float, default=ToyRoutedProjectorBankConfig.confidence_temperature
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    config = ToyRoutedProjectorBankConfig(
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
        confidence_temperature=args.confidence_temperature,
    )
    payload = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.output_md)
    return payload


if __name__ == "__main__":
    main()
