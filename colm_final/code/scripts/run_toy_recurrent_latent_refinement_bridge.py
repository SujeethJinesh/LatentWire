#!/usr/bin/env python3
"""Toy recurrent latent refinement bridge.

This bounded ablation simulates a noisy cross-model latent bridge and compares
one-shot transport against recurrent residual refinement, gated refinement,
blockwise diffusion-style denoising, and an oracle upper bound.
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


@dataclass(frozen=True)
class ToyRecurrentLatentRefinementConfig:
    seed: int = 0
    examples: int = 192
    dim: int = 24
    classes: int = 5
    block_size: int = 4
    bridge_noise: float = 0.18
    residual_noise: float = 0.09
    gate_fraction: float = 0.25
    diffusion_steps: int = 4
    diffusion_shrink: float = 0.65
    quant_bits: int = 4


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _sample_weights(rows: int, cols: int, *, generator: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    return scale * torch.randn(rows, cols, generator=generator, dtype=torch.float32) / math.sqrt(float(cols))


def _symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _blockwise_smooth(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size <= 1:
        return x
    blocks = []
    for start in range(0, x.shape[-1], block_size):
        block = x[:, start : start + block_size]
        center = block.mean(dim=-1, keepdim=True)
        left = torch.roll(block, shifts=1, dims=-1)
        right = torch.roll(block, shifts=-1, dims=-1)
        blocks.append(0.6 * block + 0.2 * left + 0.2 * right + 0.1 * center)
    return torch.cat(blocks, dim=-1)


def _make_problem(config: ToyRecurrentLatentRefinementConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    dim = int(config.dim)
    block_size = max(int(config.block_size), 1)
    block_count = max(2, math.ceil(dim / block_size))
    class_prototypes = _sample_weights(config.classes, dim, generator=gen, scale=1.35)
    block_templates = _sample_weights(block_count, dim, generator=gen, scale=0.85)
    class_head = _sample_weights(dim, config.classes, generator=gen, scale=1.1)
    reg_head = _sample_weights(dim, 1, generator=gen, scale=0.9).squeeze(-1)
    bridge_matrix = _orthogonal_matrix(dim, gen)

    class_ids = torch.randint(config.classes, (config.examples,), generator=gen)
    block_ids = torch.randint(block_count, (config.examples,), generator=gen)

    latent = (
        0.84 * class_prototypes[class_ids]
        + 0.48 * block_templates[block_ids]
        + 0.12 * torch.randn(config.examples, dim, generator=gen)
    )
    latent = latent + 0.08 * torch.roll(latent, shifts=1, dims=-1)
    latent = latent + 0.05 * torch.roll(latent, shifts=-1, dims=-1)
    latent = latent + 0.03 * torch.tanh(latent[:, : min(6, dim)]).mean(dim=-1, keepdim=True)

    source = latent @ bridge_matrix + 0.12 * torch.randn(config.examples, dim, generator=gen)
    source = source + 0.06 * torch.roll(source, shifts=1, dims=-1)

    class_logits = latent @ class_head
    if dim >= 4:
        class_logits = class_logits + 0.12 * torch.stack(
            [
                latent[:, 0] * latent[:, -1],
                latent[:, 1] * latent[:, -2],
                latent[:, 2] * latent[:, -3],
                latent[:, 3] * latent[:, -4],
            ],
            dim=-1,
        ).mean(dim=-1, keepdim=True)
    reg_target = latent @ reg_head
    if dim >= 6:
        reg_target = reg_target + 0.14 * (latent[:, :3] * latent[:, 3:6]).sum(dim=-1)
    reg_target = reg_target + 0.04 * torch.sin(latent[:, 0])

    importance = class_head.abs().mean(dim=1) + 0.7 * reg_head.abs()

    return {
        "latent": latent.float(),
        "source": source.float(),
        "class_target": class_logits.argmax(dim=-1).long(),
        "reg_target": reg_target.float(),
        "class_head": class_head.float(),
        "reg_head": reg_head.float(),
        "bridge_matrix": bridge_matrix.float(),
        "importance": importance.float(),
    }


def _task_metrics(
    latent_hat: torch.Tensor,
    latent: torch.Tensor,
    class_head: torch.Tensor,
    reg_head: torch.Tensor,
    class_target: torch.Tensor,
    reg_target: torch.Tensor,
) -> dict[str, float]:
    class_logits = latent_hat @ class_head
    if latent_hat.shape[-1] >= 4:
        class_logits = class_logits + 0.12 * torch.stack(
            [
                latent_hat[:, 0] * latent_hat[:, -1],
                latent_hat[:, 1] * latent_hat[:, -2],
                latent_hat[:, 2] * latent_hat[:, -3],
                latent_hat[:, 3] * latent_hat[:, -4],
            ],
            dim=-1,
        ).mean(dim=-1, keepdim=True)
    reg_pred = latent_hat @ reg_head
    if latent_hat.shape[-1] >= 6:
        reg_pred = reg_pred + 0.14 * (latent_hat[:, :3] * latent_hat[:, 3:6]).sum(dim=-1)
    reg_pred = reg_pred + 0.04 * torch.sin(latent_hat[:, 0])

    probs = torch.softmax(class_logits, dim=-1)
    accuracy = (class_logits.argmax(dim=-1) == class_target).float().mean()
    mse = F.mse_loss(latent_hat, latent)
    cosine = F.cosine_similarity(latent_hat, latent, dim=-1).mean()
    reg_mse = F.mse_loss(reg_pred, reg_target)
    confidence = probs.max(dim=-1).values.mean()
    return {
        "accuracy": float(accuracy.item()),
        "mse": float(mse.item()),
        "cosine": float(cosine.item()),
        "reg_mse": float(reg_mse.item()),
        "confidence": float(confidence.item()),
    }


def _trajectory_summary(trajectory: list[torch.Tensor], latent: torch.Tensor) -> dict[str, Any]:
    mse_values = [float(F.mse_loss(step, latent).item()) for step in trajectory]
    cosine_values = [float(F.cosine_similarity(step, latent, dim=-1).mean().item()) for step in trajectory]
    residual_norms = [float((step - latent).norm(dim=-1).mean().item()) for step in trajectory]
    best_step = int(min(range(len(mse_values)), key=mse_values.__getitem__))
    return {
        "trajectory_mse": mse_values,
        "trajectory_cosine": cosine_values,
        "trajectory_residual_norm": residual_norms,
        "trajectory_best_step": float(best_step),
        "trajectory_initial_mse": mse_values[0],
        "trajectory_final_mse": mse_values[-1],
        "trajectory_best_mse": min(mse_values),
        "trajectory_convergence_ratio": mse_values[-1] / max(mse_values[0], 1e-8),
    }


def _bytes_for_values(count: int, bits: float) -> float:
    return math.ceil(count * bits / 8.0)


def _compute_proxy(dim: int, steps: int, *, block_size: int) -> float:
    return float(dim * (1.0 + 0.35 * max(steps - 1, 0) + 0.1 * max(block_size - 1, 0)))


def _one_shot_bridge(problem: dict[str, torch.Tensor], config: ToyRecurrentLatentRefinementConfig) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    gen = _make_generator(config.seed + 11)
    estimate = problem["source"] @ problem["bridge_matrix"].T
    estimate = estimate + config.bridge_noise * torch.randn(estimate.shape, generator=gen, dtype=estimate.dtype)
    estimate = _symmetric_quantize(estimate, bits=int(config.quant_bits))
    bytes_estimate = _bytes_for_values(problem["latent"].shape[-1], float(config.quant_bits)) + 4.0
    compute_proxy = _compute_proxy(problem["latent"].shape[-1], 1, block_size=config.block_size)
    return estimate, [estimate], {"bytes_estimate": float(bytes_estimate), "compute_proxy": compute_proxy, "steps": 1.0}


def _two_step_residual_refinement(
    problem: dict[str, torch.Tensor], config: ToyRecurrentLatentRefinementConfig
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    gen = _make_generator(config.seed + 23)
    estimate = problem["source"] @ problem["bridge_matrix"].T
    estimate = estimate + 0.7 * config.bridge_noise * torch.randn(estimate.shape, generator=gen, dtype=estimate.dtype)
    estimate = _symmetric_quantize(estimate, bits=max(int(config.quant_bits) - 1, 2))
    refined = 0.65 * estimate + 0.35 * _blockwise_smooth(estimate, int(config.block_size))
    residual = _blockwise_smooth(refined, int(config.block_size)) - refined
    refined = refined + 0.55 * residual
    refined = refined + 0.15 * _symmetric_quantize(residual, bits=max(int(config.quant_bits) - 1, 2))
    trajectory = [estimate, refined]
    bytes_estimate = _bytes_for_values(problem["latent"].shape[-1], float(max(int(config.quant_bits) - 1, 2))) + 2.0 * _bytes_for_values(
        problem["latent"].shape[-1], 3.0
    )
    compute_proxy = _compute_proxy(problem["latent"].shape[-1], 2, block_size=config.block_size)
    return refined, trajectory, {"bytes_estimate": float(bytes_estimate), "compute_proxy": compute_proxy, "steps": 2.0}


def _gated_refinement(
    problem: dict[str, torch.Tensor], config: ToyRecurrentLatentRefinementConfig
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    estimate = problem["source"] @ problem["bridge_matrix"].T
    estimate = estimate + 0.9 * config.residual_noise * torch.roll(estimate, shifts=1, dims=-1)
    estimate = _symmetric_quantize(estimate, bits=max(int(config.quant_bits) - 1, 2))
    smooth = _blockwise_smooth(estimate, int(config.block_size))
    uncertainty = (estimate - smooth).abs() + 0.15 * problem["importance"].unsqueeze(0)
    keep = max(1, min(int(round(config.gate_fraction * estimate.shape[-1])), estimate.shape[-1]))
    keep_idx = torch.topk(uncertainty, k=keep, dim=-1).indices
    refined = smooth.clone()
    refined.scatter_(dim=-1, index=keep_idx, src=estimate.gather(dim=-1, index=keep_idx))
    refined = refined + 0.2 * (smooth - refined)
    bytes_estimate = _bytes_for_values(problem["latent"].shape[-1], float(max(int(config.quant_bits) - 1, 2))) + _bytes_for_values(
        keep, 8.0
    ) + 4.0
    compute_proxy = _compute_proxy(problem["latent"].shape[-1], 2, block_size=config.block_size)
    gate_entropy = float(torch.log(torch.tensor(float(estimate.shape[-1]) / float(keep), dtype=estimate.dtype)).item())
    return refined, [estimate, refined], {
        "bytes_estimate": float(bytes_estimate),
        "compute_proxy": compute_proxy,
        "steps": 2.0,
        "gate_entropy": gate_entropy,
        "gate_fraction_effective": float(keep / estimate.shape[-1]),
    }


def _blockwise_diffusion_denoise(
    problem: dict[str, torch.Tensor], config: ToyRecurrentLatentRefinementConfig
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    gen = _make_generator(config.seed + 37)
    source_projection = problem["source"] @ problem["bridge_matrix"].T
    estimate = source_projection + config.bridge_noise * torch.randn(source_projection.shape, generator=gen, dtype=source_projection.dtype)
    estimate = _symmetric_quantize(estimate, bits=int(config.quant_bits))
    trajectory = [estimate]
    guide = _blockwise_smooth(source_projection, int(config.block_size))
    guide = 0.75 * guide + 0.25 * source_projection
    current = estimate
    steps = max(int(config.diffusion_steps), 1)
    for step in range(steps):
        decay = float(config.diffusion_shrink) ** float(step + 1)
        current = (1.0 - decay) * current + decay * guide
        current = current + 0.12 * decay * (_blockwise_smooth(current, int(config.block_size)) - current)
        trajectory.append(current)
    bytes_estimate = _bytes_for_values(problem["latent"].shape[-1], float(config.quant_bits)) + steps * _bytes_for_values(
        int(config.block_size), 4.0
    )
    compute_proxy = _compute_proxy(problem["latent"].shape[-1], steps + 1, block_size=config.block_size)
    return current, trajectory, {"bytes_estimate": float(bytes_estimate), "compute_proxy": compute_proxy, "steps": float(steps + 1)}


def _oracle_upper_bound(problem: dict[str, torch.Tensor], config: ToyRecurrentLatentRefinementConfig) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, float]]:
    latent = problem["latent"]
    return latent, [latent], {"bytes_estimate": 0.0, "compute_proxy": 0.0, "steps": 0.0}


def run_experiment(config: ToyRecurrentLatentRefinementConfig) -> list[dict[str, Any]]:
    problem = _make_problem(config)
    methods = (
        ("one_shot_bridge", _one_shot_bridge),
        ("two_step_residual_refinement", _two_step_residual_refinement),
        ("gated_refinement", _gated_refinement),
        ("blockwise_diffusion_denoise", _blockwise_diffusion_denoise),
        ("oracle_upper_bound", _oracle_upper_bound),
    )

    one_shot_estimate, one_shot_trajectory, _ = _one_shot_bridge(problem, config)
    one_shot_errors = ((one_shot_estimate - problem["latent"]) ** 2).mean(dim=-1)

    rows: list[dict[str, Any]] = []
    for method, fn in methods:
        latent_hat, trajectory, sidecar = fn(problem, config)
        metrics = _task_metrics(
            latent_hat,
            problem["latent"],
            problem["class_head"],
            problem["reg_head"],
            problem["class_target"],
            problem["reg_target"],
        )
        summary = _trajectory_summary(trajectory, problem["latent"])
        per_example_errors = ((latent_hat - problem["latent"]) ** 2).mean(dim=-1)
        if method == "one_shot_bridge":
            help_rate = 0.0
            harm_rate = 0.0
            accuracy_delta = 0.0
            mse_delta = 0.0
            cosine_delta = 0.0
        else:
            help_rate = float((per_example_errors < one_shot_errors).float().mean().item())
            harm_rate = float((per_example_errors > one_shot_errors).float().mean().item())
            one_shot_metrics = _task_metrics(
                one_shot_estimate,
                problem["latent"],
                problem["class_head"],
                problem["reg_head"],
                problem["class_target"],
                problem["reg_target"],
            )
            accuracy_delta = metrics["accuracy"] - one_shot_metrics["accuracy"]
            mse_delta = metrics["mse"] - one_shot_metrics["mse"]
            cosine_delta = metrics["cosine"] - one_shot_metrics["cosine"]

        row: dict[str, Any] = {
            "method": method,
            "accuracy": metrics["accuracy"],
            "mse": metrics["mse"],
            "cosine": metrics["cosine"],
            "reg_mse": metrics["reg_mse"],
            "confidence": metrics["confidence"],
            "bytes_estimate": sidecar["bytes_estimate"],
            "compute_proxy": sidecar["compute_proxy"],
            "steps": sidecar["steps"],
            "help_rate": help_rate,
            "harm_rate": harm_rate,
            "net_help_rate": help_rate - harm_rate,
            "accuracy_delta_vs_one_shot": accuracy_delta,
            "mse_delta_vs_one_shot": mse_delta,
            "cosine_delta_vs_one_shot": cosine_delta,
        }
        row.update(summary)
        if "gate_entropy" in sidecar:
            row["gate_entropy"] = sidecar["gate_entropy"]
        if "gate_fraction_effective" in sidecar:
            row["gate_fraction_effective"] = sidecar["gate_fraction_effective"]
        rows.append(row)
    return rows


def write_markdown_summary(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    lines = [
        "# Toy Recurrent Latent Refinement Bridge",
        "",
        "| Method | Accuracy | MSE | Cosine | Steps | Bytes estimate | Compute proxy | Help rate | Harm rate | Trajectory MSE start | Trajectory MSE end |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {mse} | {cosine} | {steps} | {bytes_estimate} | {compute_proxy} | {help_rate} | {harm_rate} | {trajectory_initial_mse} | {trajectory_final_mse} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                mse=fmt(row["mse"]),
                cosine=fmt(row["cosine"]),
                steps=fmt(row["steps"]),
                bytes_estimate=fmt(row["bytes_estimate"]),
                compute_proxy=fmt(row["compute_proxy"]),
                help_rate=fmt(row["help_rate"]),
                harm_rate=fmt(row["harm_rate"]),
                trajectory_initial_mse=fmt(row["trajectory_initial_mse"]),
                trajectory_final_mse=fmt(row["trajectory_final_mse"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy recurrent latent refinement bridge.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=192)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--bridge-noise", type=float, default=0.18)
    parser.add_argument("--residual-noise", type=float, default=0.09)
    parser.add_argument("--gate-fraction", type=float, default=0.25)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--diffusion-shrink", type=float, default=0.65)
    parser.add_argument("--quant-bits", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyRecurrentLatentRefinementConfig(
        seed=args.seed,
        examples=args.examples,
        dim=args.dim,
        classes=args.classes,
        block_size=args.block_size,
        bridge_noise=args.bridge_noise,
        residual_noise=args.residual_noise,
        gate_fraction=args.gate_fraction,
        diffusion_steps=args.diffusion_steps,
        diffusion_shrink=args.diffusion_shrink,
        quant_bits=args.quant_bits,
    )
    rows = run_experiment(config)
    payload = {"config": asdict(config), "rows": rows}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        write_markdown_summary(rows, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
