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


@dataclass(frozen=True)
class ToyLatentRefinementConfig:
    seed: int = 0
    examples: int = 192
    dim: int = 24
    classes: int = 5
    codebook_size: int = 6
    query_bank_size: int = 4
    refinement_steps: int = 3
    gate_fraction: float = 0.25
    bridge_noise: float = 0.12
    residual_noise: float = 0.06
    soft_temperature: float = 0.75


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _sample_weights(
    rows: int,
    cols: int,
    *,
    generator: torch.Generator,
    scale: float = 1.0,
) -> torch.Tensor:
    return scale * torch.randn(rows, cols, generator=generator, dtype=torch.float32) / math.sqrt(float(cols))


def _make_problem(config: ToyLatentRefinementConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    dim = int(config.dim)
    codebook = _sample_weights(config.codebook_size, dim, generator=gen, scale=1.4)
    query_bank = _sample_weights(config.query_bank_size, dim, generator=gen, scale=0.9)
    class_head = _sample_weights(dim, config.classes, generator=gen, scale=1.2)
    reg_head = _sample_weights(dim, 1, generator=gen, scale=0.8).squeeze(-1)

    coarse_ids = torch.randint(config.codebook_size, (config.examples,), generator=gen)
    fine_ids = torch.randint(config.query_bank_size, (config.examples,), generator=gen)

    latent = (
        0.94 * codebook[coarse_ids]
        + 0.42 * query_bank[fine_ids]
        + 0.12 * torch.randn(config.examples, dim, generator=gen)
    )
    latent = latent + 0.05 * torch.tanh(latent[:, : min(4, dim)]).mean(dim=-1, keepdim=True)
    latent = latent + 0.03 * torch.roll(latent, shifts=1, dims=-1)

    class_logits = latent @ class_head
    if dim >= 4:
        class_logits = class_logits + 0.18 * torch.stack(
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
        reg_target = reg_target + 0.15 * (latent[:, :3] * latent[:, 3:6]).sum(dim=-1)
    reg_target = reg_target + 0.05 * torch.sin(latent[:, 0])

    importance = class_head.abs().mean(dim=1) + 0.7 * reg_head.abs()

    return {
        "latent": latent.float(),
        "class_target": class_logits.argmax(dim=-1).long(),
        "reg_target": reg_target.float(),
        "codebook": codebook.float(),
        "query_bank": query_bank.float(),
        "class_head": class_head.float(),
        "reg_head": reg_head.float(),
        "importance": importance.float(),
    }


def _symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _prediction_metrics(
    latent_hat: torch.Tensor,
    class_head: torch.Tensor,
    reg_head: torch.Tensor,
    class_target: torch.Tensor,
    reg_target: torch.Tensor,
) -> dict[str, float]:
    class_logits = latent_hat @ class_head
    if latent_hat.shape[-1] >= 4:
        class_logits = class_logits + 0.18 * torch.stack(
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
        reg_pred = reg_pred + 0.15 * (latent_hat[:, :3] * latent_hat[:, 3:6]).sum(dim=-1)
    reg_pred = reg_pred + 0.05 * torch.sin(latent_hat[:, 0])

    probs = torch.softmax(class_logits, dim=-1)
    entropy = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    confidence = probs.max(dim=-1).values.mean()
    accuracy = (class_logits.argmax(dim=-1) == class_target).float().mean()
    mse = F.mse_loss(reg_pred, reg_target)
    return {
        "accuracy": float(accuracy.item()),
        "mse": float(mse.item()),
        "entropy": float(entropy.item()),
        "confidence": float(confidence.item()),
    }


def _bytes_for_values(count: int, bits: float) -> float:
    return math.ceil(count * bits / 8.0)


def _method_one_shot(problem: dict[str, torch.Tensor], config: ToyLatentRefinementConfig) -> tuple[torch.Tensor, dict[str, float]]:
    gen = _make_generator(config.seed + 11)
    latent = problem["latent"]
    noisy = latent + config.bridge_noise * torch.randn(latent.shape, generator=gen, dtype=latent.dtype)
    recon = _symmetric_quantize(noisy, bits=4)
    bytes_estimate = _bytes_for_values(latent.shape[-1], 4.0) + 4.0
    return recon, {"bytes_estimate": float(bytes_estimate), "steps": 1.0}


def _method_iterative_residual(
    problem: dict[str, torch.Tensor], config: ToyLatentRefinementConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    gen = _make_generator(config.seed + 23)
    latent = problem["latent"]
    steps = max(int(config.refinement_steps), 2)
    estimate = _symmetric_quantize(latent + 0.5 * config.residual_noise * torch.randn(latent.shape, generator=gen), bits=3)
    for step in range(1, steps):
        residual = latent - estimate
        residual = residual + (config.residual_noise / float(step + 1)) * torch.randn(
            latent.shape, generator=gen, dtype=latent.dtype
        )
        estimate = estimate + _symmetric_quantize(residual, bits=3 + min(step, 2))
    bytes_estimate = _bytes_for_values(latent.shape[-1], 3.0)
    bytes_estimate += (steps - 1) * _bytes_for_values(latent.shape[-1], 3.0)
    bytes_estimate += 4.0 * steps
    return estimate, {"bytes_estimate": float(bytes_estimate), "steps": float(steps)}


def _method_gated_refinement(
    problem: dict[str, torch.Tensor], config: ToyLatentRefinementConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    latent = problem["latent"]
    importance = problem["importance"]
    keep = max(1, min(int(round(config.gate_fraction * latent.shape[-1])), latent.shape[-1]))
    scores = latent.abs() * importance.unsqueeze(0)
    keep_idx = torch.topk(scores, k=keep, dim=-1).indices

    recon = _symmetric_quantize(latent, bits=3)
    recon.scatter_(dim=-1, index=keep_idx, src=latent.gather(dim=-1, index=keep_idx))

    bytes_estimate = _bytes_for_values(latent.shape[-1] - keep, 3.0) + _bytes_for_values(keep, 8.0) + 4.0
    gate_entropy = torch.log(torch.tensor(float(latent.shape[-1]) / float(keep), dtype=latent.dtype))
    return recon, {
        "bytes_estimate": float(bytes_estimate),
        "steps": 2.0,
        "gate_entropy": float(gate_entropy.item()),
    }


def _method_soft_token_projection(
    problem: dict[str, torch.Tensor], config: ToyLatentRefinementConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    latent = problem["latent"]
    codebook = problem["codebook"]
    query_bank = problem["query_bank"]
    temp = max(float(config.soft_temperature), 1e-3)
    atoms = torch.cat([codebook, query_bank], dim=0)
    logits = (latent @ atoms.T) / (math.sqrt(float(latent.shape[-1])) * temp)
    weights = torch.softmax(logits, dim=-1)
    recon = weights @ atoms
    bytes_estimate = _bytes_for_values(atoms.shape[0], 4.0) + 4.0
    return recon, {"bytes_estimate": float(bytes_estimate), "steps": 1.0}


def _method_coarse_to_fine_query_bank(
    problem: dict[str, torch.Tensor], config: ToyLatentRefinementConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    latent = problem["latent"]
    codebook = problem["codebook"]
    query_bank = problem["query_bank"]
    temp = max(float(config.soft_temperature), 1e-3)

    coarse_logits = (latent @ codebook.T) / math.sqrt(float(latent.shape[-1]))
    coarse_ids = coarse_logits.argmax(dim=-1)
    coarse = codebook[coarse_ids]

    residual = latent - coarse
    fine_logits = (residual @ query_bank.T) / (math.sqrt(float(latent.shape[-1])) * temp)
    fine_weights = torch.softmax(fine_logits, dim=-1)
    recon = coarse + 0.6 * (fine_weights @ query_bank) + 0.6 * residual

    coarse_bits = max(1, math.ceil(math.log2(max(int(codebook.shape[0]), 2))))
    fine_bits = query_bank.shape[0] * 4.0
    bytes_estimate = _bytes_for_values(coarse_bits, 1.0) + _bytes_for_values(query_bank.shape[0], 4.0) + 4.0
    return recon, {"bytes_estimate": float(bytes_estimate), "steps": 2.0, "fine_bits": float(fine_bits)}


def run_experiment(config: ToyLatentRefinementConfig) -> list[dict[str, Any]]:
    problem = _make_problem(config)
    methods = (
        ("one_shot_noisy_bridge", _method_one_shot),
        ("iterative_residual_refinement", _method_iterative_residual),
        ("gated_refinement", _method_gated_refinement),
        ("soft_token_mixture_projection", _method_soft_token_projection),
        ("coarse_to_fine_query_bank", _method_coarse_to_fine_query_bank),
    )

    rows: list[dict[str, Any]] = []
    for method, fn in methods:
        latent_hat, sidecar = fn(problem, config)
        metrics = _prediction_metrics(
            latent_hat,
            problem["class_head"],
            problem["reg_head"],
            problem["class_target"],
            problem["reg_target"],
        )
        row = {
            "method": method,
            "accuracy": metrics["accuracy"],
            "mse": metrics["mse"],
            "entropy": metrics["entropy"],
            "confidence": metrics["confidence"],
            "bytes_estimate": sidecar["bytes_estimate"],
            "steps": sidecar["steps"],
        }
        if "gate_entropy" in sidecar:
            row["gate_entropy"] = sidecar["gate_entropy"]
        if "fine_bits" in sidecar:
            row["fine_bits"] = sidecar["fine_bits"]
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
        "# Toy Latent Refinement Bridge",
        "",
        "| Method | Accuracy | MSE | Entropy | Confidence | Bytes estimate | Steps |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {mse} | {entropy} | {confidence} | {bytes_estimate} | {steps} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                mse=fmt(row["mse"]),
                entropy=fmt(row["entropy"]),
                confidence=fmt(row["confidence"]),
                bytes_estimate=fmt(row["bytes_estimate"]),
                steps=fmt(row["steps"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact latent refinement toy.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=192)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--codebook-size", type=int, default=6)
    parser.add_argument("--query-bank-size", type=int, default=4)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--gate-fraction", type=float, default=0.25)
    parser.add_argument("--bridge-noise", type=float, default=0.12)
    parser.add_argument("--residual-noise", type=float, default=0.06)
    parser.add_argument("--soft-temperature", type=float, default=0.75)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyLatentRefinementConfig(
        seed=args.seed,
        examples=args.examples,
        dim=args.dim,
        classes=args.classes,
        codebook_size=args.codebook_size,
        query_bank_size=args.query_bank_size,
        refinement_steps=args.refinement_steps,
        gate_fraction=args.gate_fraction,
        bridge_noise=args.bridge_noise,
        residual_noise=args.residual_noise,
        soft_temperature=args.soft_temperature,
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
