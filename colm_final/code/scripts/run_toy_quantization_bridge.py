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
class ToyQuantizationConfig:
    seed: int = 0
    samples: int = 256
    dim: int = 32
    bits: int = 4
    protected_channels: int = 4
    outlier_channels: int = 4
    outlier_scale: float = 8.0
    rotations: tuple[str, ...] = ("none", "random", "hadamard")


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _normalized_hadamard(dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim < 1 or dim & (dim - 1):
        raise ValueError("Hadamard rotation requires a power-of-two dimension")
    matrix = torch.tensor([[1.0]], device=device, dtype=dtype)
    while matrix.shape[0] < dim:
        top = torch.cat([matrix, matrix], dim=1)
        bottom = torch.cat([matrix, -matrix], dim=1)
        matrix = torch.cat([top, bottom], dim=0)
    return matrix / math.sqrt(float(dim))


def _rotation_matrix(
    dim: int,
    rotation: str,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if rotation == "none":
        return torch.eye(dim, device=device, dtype=dtype)
    if rotation == "random":
        return _orthogonal_matrix(dim, _make_generator(seed)).to(device=device, dtype=dtype)
    if rotation == "hadamard":
        if dim & (dim - 1):
            return _orthogonal_matrix(dim, _make_generator(seed + 17)).to(device=device, dtype=dtype)
        hadamard = _normalized_hadamard(dim, device=device, dtype=dtype)
        signs = torch.where(
            torch.randn(dim, generator=_make_generator(seed + 31), device=device) >= 0,
            torch.ones(dim, device=device, dtype=dtype),
            -torch.ones(dim, device=device, dtype=dtype),
        )
        perm = torch.randperm(dim, generator=_make_generator(seed + 53), device=device)
        return hadamard @ torch.diag(signs)[:, perm]
    raise ValueError(f"Unknown rotation: {rotation}")


def _make_latents(config: ToyQuantizationConfig) -> tuple[torch.Tensor, torch.Tensor]:
    gen = _make_generator(config.seed)
    x = torch.randn(config.samples, config.dim, generator=gen, dtype=torch.float32)
    outlier_count = max(1, min(int(config.outlier_channels), config.dim))
    x[:, :outlier_count] = x[:, :outlier_count] * float(config.outlier_scale)
    x[:, :outlier_count] = x[:, :outlier_count] + 0.25 * torch.randn(
        config.samples, outlier_count, generator=gen, dtype=torch.float32
    )
    outlier_index = torch.arange(outlier_count, dtype=torch.long)
    return x, outlier_index


def _symmetric_quantize(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale, scale.squeeze(-1)


def _bytes_for_quantized_values(count: int, bits: int) -> int:
    return int(math.ceil(count * bits / 8.0))


def _estimate_uniform_bytes(dim: int, bits: int) -> int:
    return _bytes_for_quantized_values(dim, bits) + 4


def _estimate_protected_bytes(dim: int, bits: int, protected_channels: int) -> int:
    protected = max(1, min(int(protected_channels), max(dim - 1, 1)))
    bulk = dim - protected
    index_bytes_per_channel = max(1, math.ceil(math.log2(max(dim, 2)) / 8.0))
    return _bytes_for_quantized_values(bulk, bits) + 4 + protected * 2 + protected * index_bytes_per_channel


def _quantize_uniform(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    recon, _ = _symmetric_quantize(x, bits)
    return recon, _estimate_uniform_bytes(x.shape[-1], bits)


def _quantize_protected_outliers(
    x: torch.Tensor,
    bits: int,
    protected_channels: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    dim = x.shape[-1]
    protected = max(1, min(int(protected_channels), max(dim - 1, 1)))
    channel_energy = x.pow(2).mean(dim=0)
    protected_idx = torch.topk(channel_energy, k=protected, largest=True).indices.sort().values
    bulk_mask = torch.ones(dim, dtype=torch.bool, device=x.device)
    bulk_mask[protected_idx] = False

    recon = x.clone()
    if bulk_mask.any():
        bulk_recon, _ = _symmetric_quantize(x[:, bulk_mask], bits)
        recon[:, bulk_mask] = bulk_recon
    recon[:, protected_idx] = x[:, protected_idx].to(torch.float16).to(torch.float32)
    bytes_estimate = _estimate_protected_bytes(dim, bits, protected)
    return recon, bytes_estimate, protected_idx


def _rotate(x: torch.Tensor, rotation: str, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = _rotation_matrix(x.shape[-1], rotation, seed=seed, device=x.device, dtype=x.dtype)
    return x @ matrix, matrix


def _inverse_rotate(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return x @ matrix.T


def _cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=-1).mean().item())


def _outlier_energy_retained(original: torch.Tensor, recon: torch.Tensor, outlier_index: torch.Tensor) -> float:
    original_energy = original[:, outlier_index].pow(2).sum().clamp_min(1e-8)
    recon_energy = recon[:, outlier_index].pow(2).sum()
    return float((recon_energy / original_energy).item())


def run_experiment(config: ToyQuantizationConfig, rotations: Sequence[str] | None = None) -> list[dict[str, Any]]:
    x, outlier_index = _make_latents(config)
    rows: list[dict[str, Any]] = []
    for rotation in tuple(rotations) if rotations is not None else config.rotations:
        rotated, matrix = _rotate(x, rotation, seed=config.seed + 101)
        for quantizer in ("uniform", "protected_outlier"):
            if quantizer == "uniform":
                recon_rot, bytes_estimate = _quantize_uniform(rotated, config.bits)
            else:
                recon_rot, bytes_estimate, _ = _quantize_protected_outliers(
                    rotated,
                    config.bits,
                    config.protected_channels,
                )
            recon = _inverse_rotate(recon_rot, matrix)
            rows.append(
                {
                    "method": f"{rotation}_{quantizer}",
                    "rotation": rotation,
                    "quantizer": quantizer,
                    "bits": int(config.bits),
                    "dim": int(config.dim),
                    "samples": int(config.samples),
                    "protected_channels": int(config.protected_channels),
                    "mse": float(F.mse_loss(recon, x).item()),
                    "cosine": _cosine_mean(recon, x),
                    "outlier_energy_retained": _outlier_energy_retained(x, recon, outlier_index),
                    "bytes_estimate": float(bytes_estimate),
                }
            )
    return rows


def write_markdown_summary(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    lines = [
        "# Toy Quantization Bridge",
        "",
        "| Method | Rotation | Quantizer | MSE | Cosine | Outlier energy retained | Bytes estimate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {rotation} | {quantizer} | {mse} | {cosine} | {outlier_energy_retained} | {bytes_estimate} |".format(
                method=row["method"],
                rotation=row["rotation"],
                quantizer=row["quantizer"],
                mse=fmt(row["mse"]),
                cosine=fmt(row["cosine"]),
                outlier_energy_retained=fmt(row["outlier_energy_retained"]),
                bytes_estimate=fmt(row["bytes_estimate"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy quantization bridge ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--protected-channels", type=int, default=4)
    parser.add_argument("--outlier-channels", type=int, default=4)
    parser.add_argument("--outlier-scale", type=float, default=8.0)
    parser.add_argument(
        "--rotations",
        nargs="+",
        choices=("none", "random", "hadamard"),
        default=["none", "random", "hadamard"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = ToyQuantizationConfig(
        seed=args.seed,
        samples=args.samples,
        dim=args.dim,
        bits=args.bits,
        protected_channels=args.protected_channels,
        outlier_channels=args.outlier_channels,
        outlier_scale=args.outlier_scale,
        rotations=tuple(args.rotations),
    )
    rows = run_experiment(config, args.rotations)
    payload = {"config": asdict(config), "rows": rows}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        write_markdown_summary(rows, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
