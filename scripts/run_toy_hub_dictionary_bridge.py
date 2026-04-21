#!/usr/bin/env python3
"""Deterministic toy ablation for hub dictionaries versus pairwise bridges.

The setup creates several model families with different latent gauges.  A
pairwise bridge learns one direct map for every seen ordered family pair, while
the hub method learns one encoder and decoder per family into a shared atom
dictionary.  The held-out family is omitted from pairwise bridge training but
still receives hub adapters, testing whether hub-and-spoke alignment transfers
with O(N) adapters instead of O(N^2) bridges.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch


METHODS: tuple[str, ...] = (
    "monolithic_bridge",
    "pairwise_bridges",
    "hub_shared_dictionary",
    "held_out_family_transfer",
    "random_hub",
    "oracle",
)


@dataclass(frozen=True)
class ToyHubDictionaryBridgeConfig:
    seed: int = 0
    calibration_examples: int = 240
    test_examples: int = 240
    dim: int = 20
    atoms: int = 10
    classes: int = 5
    families: int = 6
    heldout_family: int = 5
    source_noise: float = 0.035
    target_noise: float = 0.028
    hub_snap_strength: float = 0.35
    family_scale_jitter: float = 0.18
    bridge_ridge: float = 1e-3


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
    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=-1)


def _fit_linear(source: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    x = _augment_bias(source)
    eye = torch.eye(x.shape[1], dtype=x.dtype)
    eye[-1, -1] = 0.0
    return torch.linalg.solve(x.T @ x + float(ridge) * eye, x.T @ target)


def _apply_linear(source: torch.Tensor, projector: torch.Tensor) -> torch.Tensor:
    return _augment_bias(source) @ projector


def _parameter_count(projectors: int, dim: int) -> float:
    return float(projectors * (dim + 1) * dim)


def _make_problem(config: ToyHubDictionaryBridgeConfig) -> dict[str, torch.Tensor]:
    if config.atoms < config.classes:
        raise ValueError("atoms must be >= classes")
    if config.families < 3:
        raise ValueError("families must be >= 3")
    if not 0 <= config.heldout_family < config.families:
        raise ValueError("heldout_family must index a family")

    gen = _make_generator(config.seed)
    atoms = _normalize_rows(torch.randn(config.atoms, config.dim, generator=gen)) * 3.6
    family_rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.families)], dim=0)
    family_bias = 0.55 * torch.randn(config.families, config.dim, generator=gen)
    scale_noise = config.family_scale_jitter * torch.randn(config.families, config.dim, generator=gen)
    family_scales = (1.0 + scale_noise).clamp_min(0.45)
    family_skew = 0.16 * torch.randn(config.families, config.dim, generator=gen)
    return {
        "atoms": atoms.float(),
        "family_rotations": family_rotations.float(),
        "family_bias": family_bias.float(),
        "family_scales": family_scales.float(),
        "family_skew": family_skew.float(),
    }


def _sample_hub_latents(
    config: ToyHubDictionaryBridgeConfig,
    problem: dict[str, torch.Tensor],
    *,
    count: int,
    seed_offset: int,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset)
    primary = torch.randint(config.atoms, (count,), generator=gen)
    secondary = torch.randint(config.atoms - 1, (count,), generator=gen)
    secondary = torch.where(secondary >= primary, secondary + 1, secondary)
    primary_weight = 1.05 + 0.20 * torch.rand(count, generator=gen)
    secondary_weight = 0.28 + 0.14 * torch.rand(count, generator=gen)
    hub = primary_weight.view(-1, 1) * problem["atoms"][primary]
    hub = hub + secondary_weight.view(-1, 1) * problem["atoms"][secondary]
    hub = hub + 0.06 * torch.randn(count, config.dim, generator=gen)
    labels = primary.remainder(config.classes)
    return {
        "hub": hub.float(),
        "primary_atom": primary.long(),
        "secondary_atom": secondary.long(),
        "label": labels.long(),
    }


def _family_view(
    hub: torch.Tensor,
    family: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyHubDictionaryBridgeConfig,
    *,
    generator: torch.Generator,
    noise: float,
) -> torch.Tensor:
    out = torch.empty_like(hub)
    for family_id in range(config.families):
        mask = family.eq(family_id)
        if not bool(mask.any()):
            continue
        rotated = hub[mask] @ problem["family_rotations"][family_id]
        scaled = rotated * problem["family_scales"][family_id]
        skew = float(config.source_noise) * torch.tanh(hub[mask] * problem["family_skew"][family_id])
        out[mask] = (
            scaled
            + problem["family_bias"][family_id]
            + skew
            + float(noise) * torch.randn(rotated.shape, generator=generator)
        )
    return out.float()


def _ordered_seen_pairs(config: ToyHubDictionaryBridgeConfig) -> list[tuple[int, int]]:
    seen = [idx for idx in range(config.families) if idx != config.heldout_family]
    return [(source, target) for source in seen for target in seen if source != target]


def _make_pair_examples(
    config: ToyHubDictionaryBridgeConfig,
    problem: dict[str, torch.Tensor],
    *,
    count: int,
    seed_offset: int,
    seen_only: bool,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset)
    latent = _sample_hub_latents(config, problem, count=count, seed_offset=seed_offset + 1_000)
    if seen_only:
        pairs = _ordered_seen_pairs(config)
        pair_index = torch.randint(len(pairs), (count,), generator=gen)
        source_family = torch.tensor([pairs[int(idx)][0] for idx in pair_index], dtype=torch.long)
        target_family = torch.tensor([pairs[int(idx)][1] for idx in pair_index], dtype=torch.long)
    else:
        source_family = torch.randint(config.families, (count,), generator=gen)
        target_family = torch.randint(config.families - 1, (count,), generator=gen)
        target_family = torch.where(target_family >= source_family, target_family + 1, target_family)

    source = _family_view(
        latent["hub"],
        source_family,
        problem,
        config,
        generator=_make_generator(config.seed + seed_offset + 2_000),
        noise=config.source_noise,
    )
    target = _family_view(
        latent["hub"],
        target_family,
        problem,
        config,
        generator=_make_generator(config.seed + seed_offset + 3_000),
        noise=config.target_noise,
    )
    latent.update(
        {
            "source": source,
            "target": target,
            "source_family": source_family.long(),
            "target_family": target_family.long(),
        }
    )
    return latent


def _make_family_calibration(
    config: ToyHubDictionaryBridgeConfig,
    problem: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    examples_per_family = int(math.ceil(config.calibration_examples / config.families))
    hubs = []
    views = []
    families = []
    primary = []
    for family_id in range(config.families):
        latent = _sample_hub_latents(
            config,
            problem,
            count=examples_per_family,
            seed_offset=4_000 + family_id * 97,
        )
        family = torch.full((examples_per_family,), family_id, dtype=torch.long)
        view = _family_view(
            latent["hub"],
            family,
            problem,
            config,
            generator=_make_generator(config.seed + 5_000 + family_id * 97),
            noise=config.source_noise,
        )
        hubs.append(latent["hub"])
        views.append(view)
        families.append(family)
        primary.append(latent["primary_atom"])
    return {
        "hub": torch.cat(hubs, dim=0),
        "view": torch.cat(views, dim=0),
        "family": torch.cat(families, dim=0),
        "primary_atom": torch.cat(primary, dim=0),
    }


def _fit_hub_adapters(
    family_calibration: dict[str, torch.Tensor],
    config: ToyHubDictionaryBridgeConfig,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    encoders: list[torch.Tensor] = []
    decoders: list[torch.Tensor] = []
    for family_id in range(config.families):
        mask = family_calibration["family"].eq(family_id)
        view = family_calibration["view"][mask]
        hub = family_calibration["hub"][mask]
        encoders.append(_fit_linear(view, hub, config.bridge_ridge))
        decoders.append(_fit_linear(hub, view, config.bridge_ridge))
    return encoders, decoders


def _fit_pairwise_bridges(
    pair_calibration: dict[str, torch.Tensor],
    config: ToyHubDictionaryBridgeConfig,
) -> dict[tuple[int, int], torch.Tensor]:
    bridges: dict[tuple[int, int], torch.Tensor] = {}
    for source, target in _ordered_seen_pairs(config):
        mask = pair_calibration["source_family"].eq(source) & pair_calibration["target_family"].eq(target)
        if int(mask.sum().item()) < config.dim + 1:
            continue
        bridges[(source, target)] = _fit_linear(
            pair_calibration["source"][mask],
            pair_calibration["target"][mask],
            config.bridge_ridge,
        )
    return bridges


def _fit_all(config: ToyHubDictionaryBridgeConfig, problem: dict[str, torch.Tensor]) -> dict[str, Any]:
    family_calibration = _make_family_calibration(config, problem)
    pair_calibration = _make_pair_examples(
        config,
        problem,
        count=config.calibration_examples * max(1, config.families - 1),
        seed_offset=6_000,
        seen_only=True,
    )
    monolithic = _fit_linear(pair_calibration["source"], pair_calibration["target"], config.bridge_ridge)
    pairwise = _fit_pairwise_bridges(pair_calibration, config)
    encoders, decoders = _fit_hub_adapters(family_calibration, config)

    gen = _make_generator(config.seed + 7_000)
    random_encoders = [
        0.20 * torch.randn(config.dim + 1, config.dim, generator=gen, dtype=torch.float32)
        for _ in range(config.families)
    ]
    random_decoders = [
        0.20 * torch.randn(config.dim + 1, config.dim, generator=gen, dtype=torch.float32)
        for _ in range(config.families)
    ]
    return {
        "monolithic": monolithic,
        "pairwise": pairwise,
        "encoders": encoders,
        "decoders": decoders,
        "random_encoders": random_encoders,
        "random_decoders": random_decoders,
    }


def _hub_transfer(
    test: dict[str, torch.Tensor],
    encoders: Sequence[torch.Tensor],
    decoders: Sequence[torch.Tensor],
    problem: dict[str, torch.Tensor],
    config: ToyHubDictionaryBridgeConfig,
    *,
    snap: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    hub_hat = torch.empty_like(test["hub"])
    reconstruction = torch.empty_like(test["target"])
    for source_family in range(config.families):
        source_mask = test["source_family"].eq(source_family)
        if not bool(source_mask.any()):
            continue
        hub_source = _apply_linear(test["source"][source_mask], encoders[source_family])
        hub_hat[source_mask] = hub_source

    if snap:
        atom_logits = hub_hat @ problem["atoms"].T
        top2 = torch.topk(atom_logits, k=2, dim=-1).indices
        snapped = 0.0 * hub_hat
        for slot, weight in ((0, 1.0), (1, 0.32)):
            snapped = snapped + weight * problem["atoms"][top2[:, slot]]
        snapped = snapped / 1.32
        hub_hat = (1.0 - config.hub_snap_strength) * hub_hat + config.hub_snap_strength * snapped

    for target_family in range(config.families):
        target_mask = test["target_family"].eq(target_family)
        if not bool(target_mask.any()):
            continue
        reconstruction[target_mask] = _apply_linear(hub_hat[target_mask], decoders[target_family])
    return reconstruction, hub_hat


def _monolithic_transfer(test: dict[str, torch.Tensor], projector: torch.Tensor) -> torch.Tensor:
    return _apply_linear(test["source"], projector)


def _pairwise_transfer(
    test: dict[str, torch.Tensor],
    bridges: dict[tuple[int, int], torch.Tensor],
    fallback: torch.Tensor,
    config: ToyHubDictionaryBridgeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstruction = torch.empty_like(test["target"])
    pair_seen = torch.zeros(test["target"].shape[0], dtype=torch.bool)
    fallback_reconstruction = _apply_linear(test["source"], fallback)
    for idx in range(test["target"].shape[0]):
        key = (int(test["source_family"][idx].item()), int(test["target_family"][idx].item()))
        bridge = bridges.get(key)
        if bridge is None:
            reconstruction[idx] = fallback_reconstruction[idx]
        else:
            reconstruction[idx] = _apply_linear(test["source"][idx : idx + 1], bridge).squeeze(0)
            pair_seen[idx] = True
    return reconstruction, pair_seen


def _target_to_hub(
    reconstruction: torch.Tensor,
    target_family: torch.Tensor,
    encoders: Sequence[torch.Tensor],
    config: ToyHubDictionaryBridgeConfig,
) -> torch.Tensor:
    hub = torch.empty_like(reconstruction)
    for family_id in range(config.families):
        mask = target_family.eq(family_id)
        if bool(mask.any()):
            hub[mask] = _apply_linear(reconstruction[mask], encoders[family_id])
    return hub


def _class_and_atom_predictions(hub_hat: torch.Tensor, problem: dict[str, torch.Tensor], config: ToyHubDictionaryBridgeConfig) -> tuple[torch.Tensor, torch.Tensor]:
    atom_pred = (hub_hat @ problem["atoms"].T).argmax(dim=-1).long()
    class_pred = atom_pred.remainder(config.classes)
    return class_pred, atom_pred


def _failure_tags(
    *,
    correct: torch.Tensor,
    atom_correct: torch.Tensor,
    baseline_correct: torch.Tensor,
    hub_residual: torch.Tensor,
    pairwise_residual: torch.Tensor,
    heldout_mask: torch.Tensor,
    pair_seen: torch.Tensor,
) -> dict[str, int]:
    high_hub = hub_residual > hub_residual.median()
    high_pairwise = pairwise_residual > pairwise_residual.median()
    remaining = torch.ones_like(correct, dtype=torch.bool)
    tags: dict[str, int] = {}

    def add(name: str, mask: torch.Tensor) -> None:
        nonlocal remaining
        selected = remaining & mask
        tags[name] = int(selected.sum().item())
        remaining = remaining & ~selected

    add("ok_correct", correct)
    add("heldout_pair_missing", heldout_mask & ~pair_seen & ~correct)
    add("atom_confusion", correct & ~atom_correct)
    add("lost_monolithic_correct", baseline_correct & ~correct)
    add("high_hub_residual", high_hub & ~correct)
    add("high_pairwise_residual", high_pairwise & ~correct)
    add("other", remaining)
    return tags


def _metrics_for(
    method: str,
    reconstruction: torch.Tensor,
    test: dict[str, torch.Tensor],
    problem: dict[str, torch.Tensor],
    config: ToyHubDictionaryBridgeConfig,
    *,
    baseline_correct: torch.Tensor,
    baseline_mse: torch.Tensor,
    encoders: Sequence[torch.Tensor],
    pair_seen: torch.Tensor,
    adapter_count: int,
    parameter_proxy: float,
    bytes_proxy: float,
    compute_proxy: float,
    subset_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    if subset_mask is None:
        subset_mask = torch.ones(test["target"].shape[0], dtype=torch.bool)
    if not bool(subset_mask.any()):
        raise ValueError(f"empty evaluation subset for {method}")

    reconstruction_subset = reconstruction[subset_mask]
    target_subset = test["target"][subset_mask]
    target_family_subset = test["target_family"][subset_mask]
    true_hub_subset = test["hub"][subset_mask]
    label_subset = test["label"][subset_mask]
    atom_subset = test["primary_atom"][subset_mask]
    baseline_correct_subset = baseline_correct[subset_mask]
    baseline_mse_subset = baseline_mse[subset_mask]
    pair_seen_subset = pair_seen[subset_mask]
    heldout_subset = (
        test["source_family"][subset_mask].eq(config.heldout_family)
        | test["target_family"][subset_mask].eq(config.heldout_family)
    )

    decoded_hub = _target_to_hub(reconstruction_subset, target_family_subset, encoders, config)
    class_pred, atom_pred = _class_and_atom_predictions(decoded_hub, problem, config)
    correct = class_pred.eq(label_subset)
    atom_correct = atom_pred.eq(atom_subset)
    mse_per_example = (reconstruction_subset - target_subset).pow(2).mean(dim=-1)
    hub_residual_per_example = (decoded_hub - true_hub_subset).pow(2).mean(dim=-1)

    return {
        "method": method,
        "eval_examples": int(subset_mask.sum().item()),
        "task_accuracy": float(correct.float().mean().item()),
        "mse": float(mse_per_example.mean().item()),
        "atom_recovery": float(atom_correct.float().mean().item()),
        "hub_residual": float(hub_residual_per_example.mean().item()),
        "pairwise_residual": float(mse_per_example.mean().item()),
        "heldout_fraction": float(heldout_subset.float().mean().item()),
        "pair_seen_fraction": float(pair_seen_subset.float().mean().item()),
        "adapter_count": int(adapter_count),
        "parameter_proxy": float(parameter_proxy),
        "bytes_proxy": float(bytes_proxy),
        "compute_proxy": float(compute_proxy),
        "help_vs_monolithic": float(((~baseline_correct_subset) & correct).float().mean().item()),
        "harm_vs_monolithic": float((baseline_correct_subset & ~correct).float().mean().item()),
        "mse_help_vs_monolithic": float((mse_per_example < baseline_mse_subset - 1e-8).float().mean().item()),
        "mse_harm_vs_monolithic": float((mse_per_example > baseline_mse_subset + 1e-8).float().mean().item()),
        "failure_tags": _failure_tags(
            correct=correct,
            atom_correct=atom_correct,
            baseline_correct=baseline_correct_subset,
            hub_residual=hub_residual_per_example,
            pairwise_residual=mse_per_example,
            heldout_mask=heldout_subset,
            pair_seen=pair_seen_subset,
        ),
    }


def run_experiment(config: ToyHubDictionaryBridgeConfig) -> dict[str, Any]:
    problem = _make_problem(config)
    fitted = _fit_all(config, problem)
    test = _make_pair_examples(config, problem, count=config.test_examples, seed_offset=8_000, seen_only=False)

    monolithic_recon = _monolithic_transfer(test, fitted["monolithic"])
    pairwise_recon, pair_seen = _pairwise_transfer(test, fitted["pairwise"], fitted["monolithic"], config)
    hub_recon, _ = _hub_transfer(
        test,
        fitted["encoders"],
        fitted["decoders"],
        problem,
        config,
        snap=True,
    )
    random_recon, _ = _hub_transfer(
        test,
        fitted["random_encoders"],
        fitted["random_decoders"],
        problem,
        config,
        snap=False,
    )
    oracle_recon = test["target"]

    baseline_hub = _target_to_hub(monolithic_recon, test["target_family"], fitted["encoders"], config)
    baseline_class, _ = _class_and_atom_predictions(baseline_hub, problem, config)
    baseline_correct = baseline_class.eq(test["label"])
    baseline_mse = (monolithic_recon - test["target"]).pow(2).mean(dim=-1)
    heldout_mask = test["source_family"].eq(config.heldout_family) | test["target_family"].eq(config.heldout_family)

    dim = config.dim
    hub_adapter_count = 2 * config.families
    pairwise_adapter_count = len(fitted["pairwise"])
    hub_params = _parameter_count(hub_adapter_count, dim) + float(config.atoms * dim)
    pairwise_params = _parameter_count(pairwise_adapter_count, dim)
    monolithic_params = _parameter_count(1, dim)

    rows = [
        _metrics_for(
            "monolithic_bridge",
            monolithic_recon,
            test,
            problem,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            encoders=fitted["encoders"],
            pair_seen=torch.ones(config.test_examples, dtype=torch.bool),
            adapter_count=1,
            parameter_proxy=monolithic_params,
            bytes_proxy=monolithic_params * 4.0,
            compute_proxy=float(dim * dim),
        ),
        _metrics_for(
            "pairwise_bridges",
            pairwise_recon,
            test,
            problem,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            encoders=fitted["encoders"],
            pair_seen=pair_seen,
            adapter_count=pairwise_adapter_count,
            parameter_proxy=pairwise_params,
            bytes_proxy=pairwise_params * 4.0,
            compute_proxy=float(dim * dim),
        ),
        _metrics_for(
            "hub_shared_dictionary",
            hub_recon,
            test,
            problem,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            encoders=fitted["encoders"],
            pair_seen=torch.ones(config.test_examples, dtype=torch.bool),
            adapter_count=hub_adapter_count,
            parameter_proxy=hub_params,
            bytes_proxy=hub_params * 4.0 + config.atoms * 2.0,
            compute_proxy=float(2 * dim * dim + config.atoms * dim),
        ),
        _metrics_for(
            "held_out_family_transfer",
            hub_recon,
            test,
            problem,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            encoders=fitted["encoders"],
            pair_seen=torch.ones(config.test_examples, dtype=torch.bool),
            adapter_count=hub_adapter_count,
            parameter_proxy=hub_params,
            bytes_proxy=hub_params * 4.0 + config.atoms * 2.0,
            compute_proxy=float(2 * dim * dim + config.atoms * dim),
            subset_mask=heldout_mask,
        ),
        _metrics_for(
            "random_hub",
            random_recon,
            test,
            problem,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            encoders=fitted["encoders"],
            pair_seen=torch.zeros(config.test_examples, dtype=torch.bool),
            adapter_count=hub_adapter_count,
            parameter_proxy=hub_params,
            bytes_proxy=hub_params * 4.0,
            compute_proxy=float(2 * dim * dim),
        ),
        _metrics_for(
            "oracle",
            oracle_recon,
            test,
            problem,
            config,
            baseline_correct=baseline_correct,
            baseline_mse=baseline_mse,
            encoders=fitted["encoders"],
            pair_seen=torch.ones(config.test_examples, dtype=torch.bool),
            adapter_count=0,
            parameter_proxy=0.0,
            bytes_proxy=0.0,
            compute_proxy=float(dim),
        ),
    ]

    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "seen_pair_count": pairwise_adapter_count,
        "all_ordered_pair_count": config.families * (config.families - 1),
        "hub_adapter_count": hub_adapter_count,
        "rows": rows,
        "interpretation": (
            "Pairwise bridges fit seen ordered family pairs but cannot natively transfer through the held-out family. "
            "The hub dictionary pays one encoder and decoder per family, exposes atom recovery and hub residuals, "
            "and tests whether shared route atoms can replace quadratic bridge growth."
        ),
        "sources_consulted": [
            "https://arxiv.org/abs/2602.15382",
            "https://arxiv.org/abs/2410.06981",
            "https://arxiv.org/abs/2511.03945",
            "https://arxiv.org/abs/2604.09360",
        ],
    }


def _format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = [
        "# Toy Hub Dictionary Bridge",
        "",
        "Deterministic ablation for hub-and-spoke shared dictionaries versus quadratic pairwise bridges.",
        "",
        "| Method | Examples | Accuracy | MSE | Atom recovery | Hub residual | Pairwise residual | Heldout frac | Pair seen frac | Adapters | Params | Bytes | Compute | Help | Harm | MSE help | MSE harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {method} | {eval_examples} | {task_accuracy} | {mse} | {atom_recovery} | {hub_residual} | "
            "{pairwise_residual} | {heldout_fraction} | {pair_seen_fraction} | {adapter_count} | "
            "{parameter_proxy} | {bytes_proxy} | {compute_proxy} | {help_vs_monolithic} | "
            "{harm_vs_monolithic} | {mse_help_vs_monolithic} | {mse_harm_vs_monolithic} |".format(
                **{key: _format_float(value) for key, value in row.items() if key != "failure_tags"}
            )
        )
    lines.extend(["", "## Failure Tags", ""])
    for row in payload["rows"]:
        tags = ", ".join(f"{key}={value}" for key, value in sorted(row["failure_tags"].items()))
        lines.append(f"- `{row['method']}`: {tags}")
    lines.extend(
        [
            "",
            "## Scaling",
            "",
            f"- Pairwise trained adapters: {payload['seen_pair_count']} of {payload['all_ordered_pair_count']} ordered pairs.",
            f"- Hub adapters: {payload['hub_adapter_count']} encoder/decoder adapters plus shared atom dictionary.",
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Sources Consulted",
            "",
        ]
    )
    lines.extend(f"- {source}" for source in payload["sources_consulted"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--output-md", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=ToyHubDictionaryBridgeConfig.seed)
    parser.add_argument("--calibration-examples", type=int, default=ToyHubDictionaryBridgeConfig.calibration_examples)
    parser.add_argument("--test-examples", type=int, default=ToyHubDictionaryBridgeConfig.test_examples)
    parser.add_argument("--dim", type=int, default=ToyHubDictionaryBridgeConfig.dim)
    parser.add_argument("--atoms", type=int, default=ToyHubDictionaryBridgeConfig.atoms)
    parser.add_argument("--classes", type=int, default=ToyHubDictionaryBridgeConfig.classes)
    parser.add_argument("--families", type=int, default=ToyHubDictionaryBridgeConfig.families)
    parser.add_argument("--heldout-family", type=int, default=ToyHubDictionaryBridgeConfig.heldout_family)
    parser.add_argument("--source-noise", type=float, default=ToyHubDictionaryBridgeConfig.source_noise)
    parser.add_argument("--target-noise", type=float, default=ToyHubDictionaryBridgeConfig.target_noise)
    parser.add_argument("--hub-snap-strength", type=float, default=ToyHubDictionaryBridgeConfig.hub_snap_strength)
    parser.add_argument("--family-scale-jitter", type=float, default=ToyHubDictionaryBridgeConfig.family_scale_jitter)
    parser.add_argument("--bridge-ridge", type=float, default=ToyHubDictionaryBridgeConfig.bridge_ridge)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    config = ToyHubDictionaryBridgeConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        atoms=args.atoms,
        classes=args.classes,
        families=args.families,
        heldout_family=args.heldout_family,
        source_noise=args.source_noise,
        target_noise=args.target_noise,
        hub_snap_strength=args.hub_snap_strength,
        family_scale_jitter=args.family_scale_jitter,
        bridge_ridge=args.bridge_ridge,
    )
    payload = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.output_md)
    return payload


if __name__ == "__main__":
    main()
