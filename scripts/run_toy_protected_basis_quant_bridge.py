#!/usr/bin/env python3
"""Toy protected-basis quantization bridge.

This ablation models four transport choices under a near-matched byte budget:
- uniform low-bit transport
- protected salient channels
- incoherent preprocessing via Hadamard / orthogonal rotation
- mixed-bit allocation

The synthetic task is intentionally small and deterministic. We keep a fixed
calibration set to identify salient channels, then measure how each transport
scheme preserves a downstream linear decision boundary after reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ToyProtectedBasisQuantConfig:
    seed: int = 0
    calibration_samples: int = 128
    test_samples: int = 192
    dim: int = 32
    bits_uniform: int = 4
    low_bits: int = 3
    high_bits: int = 8
    protected_channels: int = 2
    mixed_high_channels: int = 6
    outlier_channels: int = 4
    outlier_scale: float = 7.5
    signal_scale: float = 2.25
    label_noise: float = 0.15


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _make_rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _normalized_hadamard(dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim < 1 or dim & (dim - 1):
        raise ValueError("Hadamard rotation requires a power-of-two dimension")
    matrix = torch.tensor([[1.0]], device=device, dtype=dtype)
    while matrix.shape[0] < dim:
        top = torch.cat([matrix, matrix], dim=1)
        bottom = torch.cat([matrix, -matrix], dim=1)
        matrix = torch.cat([top, bottom], dim=0)
    return matrix / math.sqrt(float(dim))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _rotation_matrix(dim: int, *, seed: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, str]:
    if dim > 0 and dim & (dim - 1) == 0:
        return _normalized_hadamard(dim, device=device, dtype=dtype), "hadamard"
    return _orthogonal_matrix(dim, _make_generator(seed)).to(device=device, dtype=dtype), "orthogonal"


def _make_latents(
    *,
    samples: int,
    dim: int,
    seed: int,
    outlier_channels: int,
    outlier_scale: float,
    signal_scale: float,
    label_noise: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = _make_generator(seed)
    x = torch.randn(samples, dim, generator=gen, dtype=torch.float32)
    outlier_count = max(1, min(int(outlier_channels), int(dim)))
    x[:, :outlier_count] = x[:, :outlier_count] * float(outlier_scale)
    x[:, :outlier_count] = x[:, :outlier_count] + 0.20 * torch.randn(samples, outlier_count, generator=gen)

    probe = torch.randn(dim, generator=gen, dtype=torch.float32)
    probe[:outlier_count] = probe[:outlier_count] * float(signal_scale)
    logits = x @ probe + float(label_noise) * torch.randn(samples, generator=gen, dtype=torch.float32)
    labels = (logits >= 0).to(torch.long)
    return x, labels, probe


def _calibrate_salient_channels(x: torch.Tensor, k: int) -> torch.Tensor:
    energy = x.pow(2).mean(dim=0)
    k = max(1, min(int(k), x.shape[-1]))
    return torch.topk(energy, k=k, largest=True).indices.sort().values


def _symmetric_quantize(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale, scale.squeeze(-1)


def _bytes_for_values(count: int, bits: int) -> int:
    return int(math.ceil(count * bits / 8.0))


def _estimate_uniform_bytes(dim: int, bits: int) -> int:
    return _bytes_for_values(dim, bits) + 4


def _estimate_protected_bytes(dim: int, bits: int, protected_channels: int) -> int:
    protected = max(1, min(int(protected_channels), max(dim - 1, 1)))
    bulk = dim - protected
    index_bytes_per_channel = max(1, math.ceil(math.log2(max(dim, 2)) / 8.0))
    return _bytes_for_values(bulk, bits) + 4 + protected * 2 + protected * index_bytes_per_channel


def _estimate_mixed_bytes(dim: int, low_bits: int, high_bits: int, high_channels: int) -> int:
    high = max(1, min(int(high_channels), max(dim - 1, 1)))
    low = dim - high
    return _bytes_for_values(high, high_bits) + _bytes_for_values(low, low_bits) + 4


def _cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=-1).mean().item())


def _outlier_mass(original: torch.Tensor, recon: torch.Tensor, salient_idx: torch.Tensor) -> float:
    salient = salient_idx.to(device=original.device)
    recon_salient = recon[:, salient].pow(2).sum()
    recon_total = recon.pow(2).sum().clamp_min(1e-8)
    ratio = float((recon_salient / recon_total).item())
    return max(0.0, min(1.0, ratio))


def _quantize_uniform(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    recon, _ = _symmetric_quantize(x, bits)
    return recon, _estimate_uniform_bytes(x.shape[-1], bits)


def _quantize_protected_salient_channels(
    x: torch.Tensor,
    *,
    bits: int,
    protected_idx: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    dim = x.shape[-1]
    protected_idx = protected_idx.sort().values
    mask = torch.ones(dim, dtype=torch.bool, device=x.device)
    mask[protected_idx] = False
    recon = x.clone()
    if mask.any():
        bulk_recon, _ = _symmetric_quantize(x[:, mask], bits)
        recon[:, mask] = bulk_recon
    recon[:, protected_idx] = x[:, protected_idx].to(torch.float16).to(torch.float32)
    return recon, _estimate_protected_bytes(dim, bits, int(protected_idx.numel()))


def _quantize_mixed_bits(
    x: torch.Tensor,
    *,
    low_bits: int,
    high_bits: int,
    high_idx: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    dim = x.shape[-1]
    high_idx = high_idx.sort().values
    mask = torch.ones(dim, dtype=torch.bool, device=x.device)
    mask[high_idx] = False
    recon = x.clone()
    if mask.any():
        low_recon, _ = _symmetric_quantize(x[:, mask], low_bits)
        recon[:, mask] = low_recon
    if high_idx.numel() > 0:
        high_recon, _ = _symmetric_quantize(x[:, high_idx], high_bits)
        recon[:, high_idx] = high_recon
    return recon, _estimate_mixed_bytes(dim, low_bits, high_bits, int(high_idx.numel()))


def _apply_incoherent_preprocess(x: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    matrix, basis = _rotation_matrix(x.shape[-1], seed=seed, device=x.device, dtype=x.dtype)
    return x @ matrix, matrix, basis


def _inverse_rotate(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return x @ matrix.T


def _accuracy(x: torch.Tensor, probe: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (x @ probe >= 0).to(torch.long)
    return float((predictions == labels).float().mean().item())


def run_experiment(config: ToyProtectedBasisQuantConfig) -> dict[str, Any]:
    calibration_x, _, _ = _make_latents(
        samples=config.calibration_samples,
        dim=config.dim,
        seed=config.seed,
        outlier_channels=config.outlier_channels,
        outlier_scale=config.outlier_scale,
        signal_scale=config.signal_scale,
        label_noise=config.label_noise,
    )
    test_x, labels, probe = _make_latents(
        samples=config.test_samples,
        dim=config.dim,
        seed=config.seed + 97,
        outlier_channels=config.outlier_channels,
        outlier_scale=config.outlier_scale,
        signal_scale=config.signal_scale,
        label_noise=config.label_noise,
    )

    salient_limit = max(int(config.protected_channels), int(config.mixed_high_channels))
    salient_idx = _calibrate_salient_channels(calibration_x, salient_limit)
    protected_idx = salient_idx[: int(config.protected_channels)]
    mixed_idx = salient_idx[: int(config.mixed_high_channels)]

    rows: list[dict[str, Any]] = []
    baseline_rows: dict[str, dict[str, Any]] = {}

    uniform_recon, uniform_bytes = _quantize_uniform(test_x, config.bits_uniform)
    uniform_row = {
        "method": "uniform_low_bit",
        "basis_preprocess": "none",
        "bits_uniform": int(config.bits_uniform),
        "low_bits": int(config.low_bits),
        "high_bits": int(config.high_bits),
        "protected_channels": int(config.protected_channels),
        "mixed_high_channels": int(config.mixed_high_channels),
        "accuracy": _accuracy(uniform_recon, probe, labels),
        "mse": float(F.mse_loss(uniform_recon, test_x).item()),
        "cosine": _cosine_mean(uniform_recon, test_x),
        "outlier_mass": _outlier_mass(test_x, uniform_recon, salient_idx),
        "bytes_estimate": float(uniform_bytes),
        "byte_delta_vs_uniform": 0.0,
        "help_vs_uniform": 0.0,
        "harm_vs_uniform": 0.0,
        "salient_indices": [int(i) for i in salient_idx.tolist()],
        "selected_channels": [int(i) for i in protected_idx.tolist()],
    }
    rows.append(uniform_row)
    baseline_rows["uniform_low_bit"] = uniform_row

    protected_recon, protected_bytes = _quantize_protected_salient_channels(
        test_x,
        bits=config.low_bits,
        protected_idx=protected_idx,
    )
    protected_row = {
        "method": "protected_salient_channels",
        "basis_preprocess": "none",
        "bits_uniform": int(config.bits_uniform),
        "low_bits": int(config.low_bits),
        "high_bits": int(config.high_bits),
        "protected_channels": int(config.protected_channels),
        "mixed_high_channels": int(config.mixed_high_channels),
        "accuracy": _accuracy(protected_recon, probe, labels),
        "mse": float(F.mse_loss(protected_recon, test_x).item()),
        "cosine": _cosine_mean(protected_recon, test_x),
        "outlier_mass": _outlier_mass(test_x, protected_recon, protected_idx),
        "bytes_estimate": float(protected_bytes),
        "byte_delta_vs_uniform": float(protected_bytes - uniform_bytes),
        "help_vs_uniform": 0.0,  # filled after baseline is known
        "harm_vs_uniform": 0.0,
        "salient_indices": [int(i) for i in salient_idx.tolist()],
        "selected_channels": [int(i) for i in protected_idx.tolist()],
    }
    rows.append(protected_row)

    rotated_x, matrix, basis = _apply_incoherent_preprocess(test_x, seed=config.seed + 211)
    rotated_recon, incoherent_bytes = _quantize_uniform(rotated_x, config.bits_uniform)
    incoherent_recon = _inverse_rotate(rotated_recon, matrix)
    incoherent_row = {
        "method": "incoherent_preprocess",
        "basis_preprocess": basis,
        "bits_uniform": int(config.bits_uniform),
        "low_bits": int(config.low_bits),
        "high_bits": int(config.high_bits),
        "protected_channels": int(config.protected_channels),
        "mixed_high_channels": int(config.mixed_high_channels),
        "accuracy": _accuracy(incoherent_recon, probe, labels),
        "mse": float(F.mse_loss(incoherent_recon, test_x).item()),
        "cosine": _cosine_mean(incoherent_recon, test_x),
        "outlier_mass": _outlier_mass(test_x, incoherent_recon, salient_idx),
        "bytes_estimate": float(incoherent_bytes),
        "byte_delta_vs_uniform": float(incoherent_bytes - uniform_bytes),
        "help_vs_uniform": 0.0,
        "harm_vs_uniform": 0.0,
        "salient_indices": [int(i) for i in salient_idx.tolist()],
        "selected_channels": [int(i) for i in protected_idx.tolist()],
    }
    rows.append(incoherent_row)

    mixed_recon, mixed_bytes = _quantize_mixed_bits(
        test_x,
        low_bits=config.low_bits,
        high_bits=config.high_bits,
        high_idx=mixed_idx,
    )
    mixed_row = {
        "method": "mixed_bit_allocation",
        "basis_preprocess": "none",
        "bits_uniform": int(config.bits_uniform),
        "low_bits": int(config.low_bits),
        "high_bits": int(config.high_bits),
        "protected_channels": int(config.protected_channels),
        "mixed_high_channels": int(config.mixed_high_channels),
        "accuracy": _accuracy(mixed_recon, probe, labels),
        "mse": float(F.mse_loss(mixed_recon, test_x).item()),
        "cosine": _cosine_mean(mixed_recon, test_x),
        "outlier_mass": _outlier_mass(test_x, mixed_recon, mixed_idx),
        "bytes_estimate": float(mixed_bytes),
        "byte_delta_vs_uniform": float(mixed_bytes - uniform_bytes),
        "help_vs_uniform": 0.0,
        "harm_vs_uniform": 0.0,
        "salient_indices": [int(i) for i in salient_idx.tolist()],
        "selected_channels": [int(i) for i in mixed_idx.tolist()],
    }
    rows.append(mixed_row)

    uniform_accuracy = baseline_rows["uniform_low_bit"]["accuracy"]
    for row in rows[1:]:
        delta = row["accuracy"] - uniform_accuracy
        row["help_vs_uniform"] = float(max(0.0, delta))
        row["harm_vs_uniform"] = float(max(0.0, -delta))

    return {"config": asdict(config), "rows": rows}


def write_jsonl(rows: Sequence[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Toy Protected-Basis Quant Bridge",
        "",
        "This toy compares four transport schemes under a near-matched byte band.",
        "The calibration set identifies salient channels; the test set measures how each scheme",
        "preserves reconstruction and a downstream linear boundary.",
        "",
        "| Method | Accuracy | MSE | Cosine | Outlier mass | Bytes estimate | Help vs uniform | Harm vs uniform |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy:.4f} | {mse:.4f} | {cosine:.4f} | {outlier_mass:.4f} | {bytes_estimate:.1f} | {help_vs_uniform:.4f} | {harm_vs_uniform:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "Protected salient channels and mixed-bit allocation should recover the highest-variance",
            "coordinates under almost the same byte band, while incoherent preprocessing is the basis",
            "fix that should reduce quantization error by spreading coordinate outliers before low-bit",
            "transport. The task is intentionally small so that future bridge claims can be tied back",
            "to a specific transport choice, a specific byte budget, and a specific salient-channel",
            "selection rule.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy protected-basis quant bridge.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=192)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--bits-uniform", type=int, default=4)
    parser.add_argument("--low-bits", type=int, default=3)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--protected-channels", type=int, default=2)
    parser.add_argument("--mixed-high-channels", type=int, default=6)
    parser.add_argument("--outlier-channels", type=int, default=4)
    parser.add_argument("--outlier-scale", type=float, default=7.5)
    parser.add_argument("--signal-scale", type=float, default=2.25)
    parser.add_argument("--label-noise", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyProtectedBasisQuantConfig(
        seed=args.seed,
        calibration_samples=args.calibration_samples,
        test_samples=args.test_samples,
        dim=args.dim,
        bits_uniform=args.bits_uniform,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
        protected_channels=args.protected_channels,
        mixed_high_channels=args.mixed_high_channels,
        outlier_channels=args.outlier_channels,
        outlier_scale=args.outlier_scale,
        signal_scale=args.signal_scale,
        label_noise=args.label_noise,
    )
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    write_jsonl(payload["rows"], output)
    write_markdown_summary(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
