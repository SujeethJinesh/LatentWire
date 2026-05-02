#!/usr/bin/env python3
"""Deterministic toy ablation for iterative latent repair after transfer.

The setup simulates a source model sending a distorted latent to a target model.
Target-side refinement can spend extra compute to denoise the transferred
latent before the task head consumes it.  The ablation separates fixed-step,
confidence-gated, noisy/diffusion-style, and oracle repair paths.
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
    "one_pass_bridge",
    "fixed_2_step_refinement",
    "fixed_4_step_refinement",
    "confidence_gated_refinement",
    "noisy_diffusion_refinement",
    "oracle_refinement",
)


@dataclass(frozen=True)
class ToyIterativeLatentRefinementConfig:
    seed: int = 0
    examples: int = 160
    dim: int = 24
    classes: int = 5
    styles: int = 7
    quant_bits: int = 4
    bridge_noise: float = 0.20
    source_noise: float = 0.08
    refinement_rate: float = 0.46
    gate_threshold: float = 0.66
    diffusion_steps: int = 4
    diffusion_noise: float = 0.07
    oracle_fraction: float = 0.35


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _sample_weights(rows: int, cols: int, *, generator: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    return scale * torch.randn(rows, cols, generator=generator, dtype=torch.float32) / math.sqrt(float(cols))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _bytes_for_values(count: int, bits: float) -> float:
    return math.ceil(float(count) * float(bits) / 8.0)


def _compute_proxy(dim: int, steps: int, *, gated_fraction: float = 1.0) -> float:
    base = float(dim)
    repair = float(dim) * max(float(steps - 1), 0.0) * 0.38 * float(gated_fraction)
    return base + repair


def _local_smooth(x: torch.Tensor) -> torch.Tensor:
    return 0.55 * x + 0.25 * torch.roll(x, shifts=1, dims=-1) + 0.20 * torch.roll(x, shifts=-1, dims=-1)


def _make_problem(config: ToyIterativeLatentRefinementConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    dim = int(config.dim)
    class_prototypes = _sample_weights(config.classes, dim, generator=gen, scale=3.2)
    style_atoms = _sample_weights(config.styles, dim, generator=gen, scale=1.45)
    class_head = 0.55 * class_prototypes.T + _sample_weights(dim, config.classes, generator=gen, scale=1.15)
    reg_head = _sample_weights(dim, 1, generator=gen, scale=0.9).squeeze(-1)
    bridge_matrix = _orthogonal_matrix(dim, gen)

    class_ids = torch.randint(config.classes, (config.examples,), generator=gen)
    style_ids = torch.randint(config.styles, (config.examples,), generator=gen)
    latent = (
        class_prototypes[class_ids]
        + 0.52 * style_atoms[style_ids]
        + 0.13 * torch.randn(config.examples, dim, generator=gen)
    )
    latent = latent + 0.10 * _local_smooth(latent)
    latent = latent + 0.04 * torch.tanh(latent[:, : min(6, dim)]).mean(dim=-1, keepdim=True)

    source = latent @ bridge_matrix + config.source_noise * torch.randn(config.examples, dim, generator=gen)
    source = source + 0.05 * torch.roll(source, shifts=2, dims=-1)
    reg_target = latent @ reg_head
    if dim >= 6:
        reg_target = reg_target + 0.10 * (latent[:, :3] * latent[:, 3:6]).sum(dim=-1)
    reg_target = reg_target + 0.03 * torch.sin(latent[:, 0])
    class_logits = latent @ class_head
    if dim >= 4:
        class_logits = class_logits + 0.08 * torch.stack(
            [
                latent[:, 0] * latent[:, -1],
                latent[:, 1] * latent[:, -2],
                latent[:, 2] * latent[:, -3],
                latent[:, 3] * latent[:, -4],
            ],
            dim=-1,
        ).mean(dim=-1, keepdim=True)

    return {
        "latent": latent.float(),
        "source": source.float(),
        "class_target": class_logits.argmax(dim=-1).long(),
        "style_target": style_ids.long(),
        "class_prototypes": class_prototypes.float(),
        "style_atoms": style_atoms.float(),
        "class_head": class_head.float(),
        "reg_head": reg_head.float(),
        "bridge_matrix": bridge_matrix.float(),
        "reg_target": reg_target.float(),
    }


def _initial_bridge(problem: dict[str, torch.Tensor], config: ToyIterativeLatentRefinementConfig) -> torch.Tensor:
    gen = _make_generator(config.seed + 17)
    estimate = problem["source"] @ problem["bridge_matrix"].T
    estimate = 0.84 * estimate + 0.16 * _local_smooth(estimate)
    estimate = estimate + config.bridge_noise * torch.randn(estimate.shape, generator=gen, dtype=estimate.dtype)
    return _symmetric_quantize(estimate, int(config.quant_bits))


def _logits(latent_hat: torch.Tensor, problem: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = latent_hat @ problem["class_head"]
    if latent_hat.shape[-1] >= 4:
        logits = logits + 0.08 * torch.stack(
            [
                latent_hat[:, 0] * latent_hat[:, -1],
                latent_hat[:, 1] * latent_hat[:, -2],
                latent_hat[:, 2] * latent_hat[:, -3],
                latent_hat[:, 3] * latent_hat[:, -4],
            ],
            dim=-1,
        ).mean(dim=-1, keepdim=True)
    return logits


def _confidence(latent_hat: torch.Tensor, problem: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.softmax(_logits(latent_hat, problem), dim=-1).max(dim=-1).values


def _prototype_denoise(current: torch.Tensor, problem: dict[str, torch.Tensor], rate: float) -> torch.Tensor:
    logits = _logits(current, problem)
    class_weights = torch.softmax(logits / 1.3, dim=-1)
    class_anchor = class_weights @ problem["class_prototypes"]
    style_logits = (current @ problem["style_atoms"].T) / math.sqrt(float(current.shape[-1]))
    style_anchor = torch.softmax(style_logits / 1.1, dim=-1) @ problem["style_atoms"]
    manifold_target = class_anchor + 0.52 * style_anchor
    residual_target = 0.72 * manifold_target + 0.28 * _local_smooth(current)
    return current + float(rate) * (residual_target - current)


def _fixed_refinement(
    start: torch.Tensor, problem: dict[str, torch.Tensor], config: ToyIterativeLatentRefinementConfig, *, steps: int
) -> list[torch.Tensor]:
    trajectory = [start]
    current = start
    for step in range(max(int(steps) - 1, 0)):
        rate = float(config.refinement_rate) / math.sqrt(float(step + 1))
        current = _prototype_denoise(current, problem, rate)
        trajectory.append(current)
    return trajectory


def _confidence_gated_refinement(
    start: torch.Tensor, problem: dict[str, torch.Tensor], config: ToyIterativeLatentRefinementConfig
) -> tuple[list[torch.Tensor], torch.Tensor]:
    confidence = _confidence(start, problem)
    gate = confidence < float(config.gate_threshold)
    repaired = _prototype_denoise(start, problem, float(config.refinement_rate))
    repaired = _prototype_denoise(repaired, problem, 0.65 * float(config.refinement_rate))
    current = torch.where(gate.unsqueeze(-1), repaired, start)
    return [start, current], gate


def _noisy_diffusion_refinement(
    start: torch.Tensor, problem: dict[str, torch.Tensor], config: ToyIterativeLatentRefinementConfig
) -> list[torch.Tensor]:
    gen = _make_generator(config.seed + 41)
    trajectory = [start]
    current = start + config.diffusion_noise * torch.randn(start.shape, generator=gen, dtype=start.dtype)
    steps = max(int(config.diffusion_steps), 1)
    for step in range(steps):
        denoised = _prototype_denoise(current, problem, 0.48 * float(config.refinement_rate))
        schedule = 0.55 / math.sqrt(float(step + 1))
        anchor = 0.12 / float(step + 2)
        current = current + schedule * (denoised - current)
        current = (1.0 - anchor) * current + anchor * start
        trajectory.append(current)
    return trajectory


def _oracle_refinement(
    start: torch.Tensor, problem: dict[str, torch.Tensor], config: ToyIterativeLatentRefinementConfig
) -> list[torch.Tensor]:
    latent = problem["latent"]
    residual = latent - start
    dim = latent.shape[-1]
    keep = max(1, min(int(round(float(config.oracle_fraction) * dim)), dim))
    per_dim_error = residual.abs()
    keep_idx = torch.topk(per_dim_error, k=keep, dim=-1).indices
    patch = torch.zeros_like(residual)
    patch.scatter_(dim=-1, index=keep_idx, src=residual.gather(dim=-1, index=keep_idx))
    repaired = start + patch
    repaired = repaired + 0.35 * (latent - repaired)
    return [start, repaired]


def _regression_prediction(latent_hat: torch.Tensor, problem: dict[str, torch.Tensor]) -> torch.Tensor:
    reg_pred = latent_hat @ problem["reg_head"]
    if latent_hat.shape[-1] >= 6:
        reg_pred = reg_pred + 0.10 * (latent_hat[:, :3] * latent_hat[:, 3:6]).sum(dim=-1)
    return reg_pred + 0.03 * torch.sin(latent_hat[:, 0])


def _ece(confidence: torch.Tensor, correct: torch.Tensor, *, bins: int = 5) -> float:
    total = max(int(confidence.numel()), 1)
    error = torch.tensor(0.0, dtype=confidence.dtype)
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if not bool(mask.any()):
            continue
        gap = (confidence[mask].mean() - correct[mask].float().mean()).abs()
        error = error + (mask.float().mean() * gap)
    return float(error.item() * total / total)


def _failure_reasons(
    start: torch.Tensor,
    final: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyIterativeLatentRefinementConfig,
) -> dict[str, int]:
    target = problem["class_target"]
    start_pred = _logits(start, problem).argmax(dim=-1)
    final_logits = _logits(final, problem)
    final_pred = final_logits.argmax(dim=-1)
    confidence = torch.softmax(final_logits, dim=-1).max(dim=-1).values
    start_mse = (start - problem["latent"]).pow(2).mean(dim=-1)
    final_mse = (final - problem["latent"]).pow(2).mean(dim=-1)
    improved = final_mse < start_mse - 1e-8
    harmed = final_mse > start_mse + 1e-8

    remaining = torch.ones_like(target, dtype=torch.bool)
    reasons: dict[str, int] = {}

    def add_reason(name: str, mask: torch.Tensor) -> None:
        nonlocal remaining
        selected = remaining & mask
        reasons[name] = int(selected.sum().item())
        remaining = remaining & ~selected

    add_reason("ok_correct_improved", (final_pred == target) & improved)
    add_reason("unchanged_correct", (final_pred == target) & ~improved & ~harmed)
    add_reason("over_refined_harm", harmed & (final_pred != start_pred))
    add_reason("mse_improved_but_task_wrong", improved & (final_pred != target))
    add_reason("low_confidence_unrepaired", (final_pred != target) & (confidence < config.gate_threshold))
    add_reason("wrong_high_confidence", (final_pred != target) & (confidence >= config.gate_threshold))
    reasons["other"] = int(remaining.sum().item())
    return reasons


def _metrics_for(
    method: str,
    trajectory: list[torch.Tensor],
    problem: dict[str, torch.Tensor],
    config: ToyIterativeLatentRefinementConfig,
    *,
    bytes_proxy: float,
    compute_proxy: float,
    gate: torch.Tensor | None = None,
) -> dict[str, Any]:
    start = trajectory[0]
    final = trajectory[-1]
    logits = _logits(final, problem)
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    pred = logits.argmax(dim=-1)
    correct = pred == problem["class_target"]
    start_correct = _logits(start, problem).argmax(dim=-1) == problem["class_target"]
    latent_mse = (final - problem["latent"]).pow(2).mean(dim=-1)
    start_mse = (start - problem["latent"]).pow(2).mean(dim=-1)
    reg_mse = F.mse_loss(_regression_prediction(final, problem), problem["reg_target"])
    trajectory_mse = [float(F.mse_loss(step, problem["latent"]).item()) for step in trajectory]

    row: dict[str, Any] = {
        "method": method,
        "task_accuracy": float(correct.float().mean().item()),
        "mse": float(latent_mse.mean().item()),
        "regression_mse": float(reg_mse.item()),
        "refinement_steps": float(len(trajectory)),
        "compute_proxy": float(compute_proxy),
        "bytes_proxy": float(bytes_proxy),
        "help_rate": float(((~start_correct) & correct).float().mean().item()),
        "harm_rate": float((start_correct & ~correct).float().mean().item()),
        "mse_help_rate": float((latent_mse < start_mse - 1e-8).float().mean().item()),
        "mse_harm_rate": float((latent_mse > start_mse + 1e-8).float().mean().item()),
        "mean_confidence": float(confidence.mean().item()),
        "confidence_ece": _ece(confidence, correct),
        "trajectory_mse": trajectory_mse,
        "trajectory_best_mse": min(trajectory_mse),
        "trajectory_convergence_ratio": trajectory_mse[-1] / max(trajectory_mse[0], 1e-8),
        "failure_reasons": _failure_reasons(start, final, problem, config),
    }
    if gate is not None:
        row["gate_fraction"] = float(gate.float().mean().item())
        row["gated_examples"] = int(gate.sum().item())
    return row


def run_experiment(config: ToyIterativeLatentRefinementConfig) -> dict[str, Any]:
    problem = _make_problem(config)
    start = _initial_bridge(problem, config)
    dim = int(config.dim)
    quant_bits = float(config.quant_bits)
    base_bytes = _bytes_for_values(dim, quant_bits) + 4.0

    rows: list[dict[str, Any]] = []
    rows.append(
        _metrics_for(
            "one_pass_bridge",
            [start],
            problem,
            config,
            bytes_proxy=base_bytes,
            compute_proxy=_compute_proxy(dim, 1),
        )
    )

    for steps, name in ((2, "fixed_2_step_refinement"), (4, "fixed_4_step_refinement")):
        trajectory = _fixed_refinement(start, problem, config, steps=steps)
        bytes_proxy = base_bytes + (steps - 1) * _bytes_for_values(dim, 3.0)
        rows.append(
            _metrics_for(
                name,
                trajectory,
                problem,
                config,
                bytes_proxy=bytes_proxy,
                compute_proxy=_compute_proxy(dim, steps),
            )
        )

    gated_trajectory, gate = _confidence_gated_refinement(start, problem, config)
    gate_fraction = float(gate.float().mean().item())
    rows.append(
        _metrics_for(
            "confidence_gated_refinement",
            gated_trajectory,
            problem,
            config,
            bytes_proxy=base_bytes + gate_fraction * _bytes_for_values(dim, 6.0),
            compute_proxy=_compute_proxy(dim, 2, gated_fraction=gate_fraction),
            gate=gate,
        )
    )

    diffusion_trajectory = _noisy_diffusion_refinement(start, problem, config)
    rows.append(
        _metrics_for(
            "noisy_diffusion_refinement",
            diffusion_trajectory,
            problem,
            config,
            bytes_proxy=base_bytes + int(config.diffusion_steps) * _bytes_for_values(dim, 2.0),
            compute_proxy=_compute_proxy(dim, len(diffusion_trajectory)),
        )
    )

    oracle_trajectory = _oracle_refinement(start, problem, config)
    keep = max(1, min(int(round(float(config.oracle_fraction) * dim)), dim))
    rows.append(
        _metrics_for(
            "oracle_refinement",
            oracle_trajectory,
            problem,
            config,
            bytes_proxy=base_bytes + _bytes_for_values(keep, 8.0),
            compute_proxy=_compute_proxy(dim, 2, gated_fraction=keep / dim),
        )
    )

    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
        "interpretation": (
            "Fixed target-side refinement tests whether repeated manifold repair helps after transfer; "
            "confidence gating measures selective compute; noisy diffusion adds stochastic-denoising pressure; "
            "oracle refinement upper-bounds sparse residual repair."
        ),
    }


def _format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = [
        "# Toy Iterative Latent Refinement",
        "",
        "Deterministic ablation for target-side latent repair after noisy cross-model transfer.",
        "",
        "| Method | Accuracy | MSE | Reg MSE | Steps | Compute | Bytes | Help | Harm | MSE Help | MSE Harm | Confidence | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {method} | {task_accuracy} | {mse} | {regression_mse} | {refinement_steps} | "
            "{compute_proxy} | {bytes_proxy} | {help_rate} | {harm_rate} | {mse_help_rate} | "
            "{mse_harm_rate} | {mean_confidence} | {confidence_ece} |".format(
                **{key: _format_float(value) for key, value in row.items() if key != "failure_reasons"}
            )
        )
    lines.extend(["", "## Failure Reasons", ""])
    for row in payload["rows"]:
        reasons = ", ".join(f"{key}={value}" for key, value in sorted(row["failure_reasons"].items()))
        lines.append(f"- `{row['method']}`: {reasons}")
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--output-md", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=ToyIterativeLatentRefinementConfig.seed)
    parser.add_argument("--examples", type=int, default=ToyIterativeLatentRefinementConfig.examples)
    parser.add_argument("--dim", type=int, default=ToyIterativeLatentRefinementConfig.dim)
    parser.add_argument("--classes", type=int, default=ToyIterativeLatentRefinementConfig.classes)
    parser.add_argument("--styles", type=int, default=ToyIterativeLatentRefinementConfig.styles)
    parser.add_argument("--quant-bits", type=int, default=ToyIterativeLatentRefinementConfig.quant_bits)
    parser.add_argument("--bridge-noise", type=float, default=ToyIterativeLatentRefinementConfig.bridge_noise)
    parser.add_argument("--source-noise", type=float, default=ToyIterativeLatentRefinementConfig.source_noise)
    parser.add_argument("--refinement-rate", type=float, default=ToyIterativeLatentRefinementConfig.refinement_rate)
    parser.add_argument("--gate-threshold", type=float, default=ToyIterativeLatentRefinementConfig.gate_threshold)
    parser.add_argument("--diffusion-steps", type=int, default=ToyIterativeLatentRefinementConfig.diffusion_steps)
    parser.add_argument("--diffusion-noise", type=float, default=ToyIterativeLatentRefinementConfig.diffusion_noise)
    parser.add_argument("--oracle-fraction", type=float, default=ToyIterativeLatentRefinementConfig.oracle_fraction)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    config = ToyIterativeLatentRefinementConfig(
        seed=args.seed,
        examples=args.examples,
        dim=args.dim,
        classes=args.classes,
        styles=args.styles,
        quant_bits=args.quant_bits,
        bridge_noise=args.bridge_noise,
        source_noise=args.source_noise,
        refinement_rate=args.refinement_rate,
        gate_threshold=args.gate_threshold,
        diffusion_steps=args.diffusion_steps,
        diffusion_noise=args.diffusion_noise,
        oracle_fraction=args.oracle_fraction,
    )
    payload = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.output_md)
    return payload


if __name__ == "__main__":
    main()
