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
class ToyActivationAwareAtomQuantConfig:
    seed: int = 0
    calibration_samples: int = 160
    test_samples: int = 192
    atoms: int = 32
    signal_atoms: int = 6
    outlier_atoms: int = 4
    protected_atoms: int = 8
    low_bits: int = 3
    high_bits: int = 8
    signal_scale: float = 3.25
    outlier_scale: float = 8.5
    label_noise: float = 0.18
    activation_spike: float = 1.4


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _bytes_for_values(count: int, bits: int) -> int:
    return int(math.ceil(count * bits / 8.0))


def _estimate_uniform_bytes(atoms: int, bits: int) -> int:
    return _bytes_for_values(atoms, bits) + 4


def _estimate_mixed_bytes(atoms: int, bits_low: int, bits_high: int, protected_atoms: int) -> int:
    protected = max(0, min(int(protected_atoms), int(atoms)))
    low = atoms - protected
    index_bytes_per_atom = max(1, math.ceil(math.log2(max(atoms, 2)) / 8.0))
    return _bytes_for_values(low, bits_low) + _bytes_for_values(protected, bits_high) + 4 + protected * index_bytes_per_atom


def _symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _select_topk(values: torch.Tensor, k: int) -> torch.Tensor:
    k = max(0, min(int(k), int(values.numel())))
    if k == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    return torch.topk(values, k=k, largest=True).indices.sort().values


def _make_latents(
    *,
    samples: int,
    config: ToyActivationAwareAtomQuantConfig,
    seed_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset)
    x = torch.randn(samples, config.atoms, generator=gen, dtype=torch.float32)

    signal_atoms = max(1, min(int(config.signal_atoms), int(config.atoms)))
    outlier_atoms = max(1, min(int(config.outlier_atoms), max(int(config.atoms) - signal_atoms, 1)))

    shared_signal = torch.randn(samples, 1, generator=gen, dtype=torch.float32)
    x[:, :signal_atoms] = (
        0.65 * x[:, :signal_atoms] + float(config.signal_scale) * shared_signal * torch.linspace(
            1.0, 0.35, signal_atoms, dtype=torch.float32
        )
    )

    outlier_noise = torch.randn(samples, outlier_atoms, generator=gen, dtype=torch.float32)
    x[:, signal_atoms : signal_atoms + outlier_atoms] = (
        float(config.outlier_scale) * outlier_noise
        + float(config.activation_spike) * torch.sign(outlier_noise) * outlier_noise.abs().pow(1.15)
    )

    if signal_atoms + outlier_atoms < config.atoms:
        x[:, signal_atoms + outlier_atoms :] = 0.55 * x[:, signal_atoms + outlier_atoms :]

    true_w = torch.randn(config.atoms, generator=gen, dtype=torch.float32) * 0.25
    true_w[:signal_atoms] += torch.linspace(2.6, 1.3, signal_atoms, dtype=torch.float32)
    true_w[signal_atoms : signal_atoms + outlier_atoms] += torch.linspace(
        1.7, 0.9, outlier_atoms, dtype=torch.float32
    )

    logits = x @ true_w + float(config.label_noise) * torch.randn(samples, generator=gen, dtype=torch.float32)
    labels = (logits >= 0).to(torch.long)
    return x, labels, true_w


def _cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=-1).mean().item())


def _row_metrics(
    *,
    method: str,
    config: ToyActivationAwareAtomQuantConfig,
    full_preds: torch.Tensor,
    recon: torch.Tensor,
    test_x: torch.Tensor,
    true_w: torch.Tensor,
    labels: torch.Tensor,
    selected_atoms: torch.Tensor,
    oracle_atoms: torch.Tensor,
    outlier_atoms: torch.Tensor,
    bytes_proxy: int,
    bit_budget: float,
) -> dict[str, Any]:
    preds = (recon @ true_w >= 0).to(torch.long)
    full_correct = full_preds.eq(labels)
    method_correct = preds.eq(labels)
    help_vs_full = float((method_correct & ~full_correct).float().mean().item())
    harm_vs_full = float((~method_correct & full_correct).float().mean().item())

    selected = set(int(i) for i in selected_atoms.tolist())
    oracle = set(int(i) for i in oracle_atoms.tolist())
    outliers = set(int(i) for i in outlier_atoms.tolist())

    top_atom_preservation = 1.0 if len(oracle) == 0 else len(selected & oracle) / float(len(oracle))
    outlier_protected = 1.0 if len(outliers) == 0 else len(selected & outliers) / float(len(outliers))
    protected_rate = float(len(selected) / max(int(config.atoms), 1))

    return {
        "method": method,
        "seed": int(config.seed),
        "atoms": int(config.atoms),
        "signal_atoms": int(config.signal_atoms),
        "outlier_atoms": int(config.outlier_atoms),
        "protected_atoms": int(selected_atoms.numel()),
        "low_bits": int(config.low_bits),
        "high_bits": int(config.high_bits),
        "bit_budget": float(bit_budget),
        "bytes_proxy": float(bytes_proxy),
        "accuracy": float(method_correct.float().mean().item()),
        "mse": float(F.mse_loss(recon, test_x).item()),
        "cosine": _cosine_mean(recon, test_x),
        "protected_rate": protected_rate,
        "outlier_protected_rate": float(outlier_protected),
        "top_atom_preservation_rate": float(top_atom_preservation),
        "help_vs_full_precision": help_vs_full,
        "harm_vs_full_precision": harm_vs_full,
        "selected_atoms": [int(i) for i in selected_atoms.tolist()],
        "oracle_atoms": [int(i) for i in oracle_atoms.tolist()],
        "outlier_atom_indices": [int(i) for i in outlier_atoms.tolist()],
    }


def run_experiment(config: ToyActivationAwareAtomQuantConfig) -> dict[str, Any]:
    calibration_x, _, _ = _make_latents(samples=config.calibration_samples, config=config, seed_offset=0)
    test_x, labels, true_w = _make_latents(samples=config.test_samples, config=config, seed_offset=9_701)

    signal_atoms = max(1, min(int(config.signal_atoms), int(config.atoms)))
    outlier_atoms = max(1, min(int(config.outlier_atoms), max(int(config.atoms) - signal_atoms, 1)))
    true_outlier_atoms = torch.arange(signal_atoms, signal_atoms + outlier_atoms, dtype=torch.long)

    calibration_energy = calibration_x.pow(2).mean(dim=0)
    calibration_peak = calibration_x.abs().amax(dim=0)
    oracle_score = calibration_x.std(dim=0, unbiased=False) * torch.abs(true_w)
    random_atoms = torch.randperm(config.atoms, generator=_make_generator(config.seed + 17))[: int(config.protected_atoms)].sort().values
    activation_atoms = _select_topk(calibration_energy, config.protected_atoms)
    outlier_atoms_selected = _select_topk(calibration_peak, config.protected_atoms)
    oracle_atoms = _select_topk(oracle_score, config.protected_atoms)

    rows: list[dict[str, Any]] = []

    full_preds = (test_x @ true_w >= 0).to(torch.long)
    rows.append(
        _row_metrics(
            method="full_precision",
            config=config,
            full_preds=full_preds,
            recon=test_x,
            test_x=test_x,
            true_w=true_w,
            labels=labels,
            selected_atoms=torch.arange(config.atoms, dtype=torch.long),
            oracle_atoms=oracle_atoms,
            outlier_atoms=true_outlier_atoms,
            bytes_proxy=_estimate_uniform_bytes(config.atoms, 16),
            bit_budget=16.0,
        )
    )

    uniform_recon = _symmetric_quantize(test_x, config.low_bits)
    rows.append(
        _row_metrics(
            method="uniform_low_bit",
            config=config,
            full_preds=full_preds,
            recon=uniform_recon,
            test_x=test_x,
            true_w=true_w,
            labels=labels,
            selected_atoms=torch.empty(0, dtype=torch.long),
            oracle_atoms=oracle_atoms,
            outlier_atoms=true_outlier_atoms,
            bytes_proxy=_estimate_uniform_bytes(config.atoms, config.low_bits),
            bit_budget=float(config.low_bits),
        )
    )

    methods: list[tuple[str, torch.Tensor]] = [
        ("random_mixed_precision", random_atoms),
        ("activation_aware_mixed_precision", activation_atoms),
        ("protected_outlier_mixed_precision", outlier_atoms_selected),
        ("oracle_mixed_precision", oracle_atoms),
    ]
    for method, selected_atoms in methods:
        recon = test_x.clone()
        mask = torch.ones(config.atoms, dtype=torch.bool)
        mask[selected_atoms] = False
        if mask.any():
            recon[:, mask] = _symmetric_quantize(test_x[:, mask], config.low_bits)
        if selected_atoms.numel() > 0:
            recon[:, selected_atoms] = _symmetric_quantize(test_x[:, selected_atoms], config.high_bits)
        rows.append(
            _row_metrics(
                method=method,
                config=config,
                full_preds=full_preds,
                recon=recon,
                test_x=test_x,
                true_w=true_w,
                labels=labels,
                selected_atoms=selected_atoms,
                oracle_atoms=oracle_atoms,
                outlier_atoms=true_outlier_atoms,
                bytes_proxy=_estimate_mixed_bytes(config.atoms, config.low_bits, config.high_bits, int(selected_atoms.numel())),
                bit_budget=float(
                    (int(selected_atoms.numel()) * config.high_bits + (config.atoms - int(selected_atoms.numel())) * config.low_bits)
                    / float(config.atoms)
                ),
            )
        )

    return {"config": asdict(config), "rows": rows}


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Toy Activation-Aware Atom Quant",
        "",
        "| Method | Acc | MSE | Cosine | Bit budget | Bytes proxy | Protected rate | Outlier protected | Top-atom preservation | Help vs full | Harm vs full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy:.4f} | {mse:.4f} | {cosine:.4f} | {bit_budget:.4f} | {bytes_proxy:.1f} | {protected_rate:.4f} | {outlier_protected_rate:.4f} | {top_atom_preservation_rate:.4f} | {help_vs_full_precision:.4f} | {harm_vs_full_precision:.4f} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy activation-aware atom quantization ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=160)
    parser.add_argument("--test-samples", type=int, default=192)
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--signal-atoms", type=int, default=6)
    parser.add_argument("--outlier-atoms", type=int, default=4)
    parser.add_argument("--protected-atoms", type=int, default=8)
    parser.add_argument("--low-bits", type=int, default=3)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--signal-scale", type=float, default=3.25)
    parser.add_argument("--outlier-scale", type=float, default=8.5)
    parser.add_argument("--label-noise", type=float, default=0.18)
    parser.add_argument("--activation-spike", type=float, default=1.4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyActivationAwareAtomQuantConfig(
        seed=args.seed,
        calibration_samples=args.calibration_samples,
        test_samples=args.test_samples,
        atoms=args.atoms,
        signal_atoms=args.signal_atoms,
        outlier_atoms=args.outlier_atoms,
        protected_atoms=args.protected_atoms,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
        signal_scale=args.signal_scale,
        outlier_scale=args.outlier_scale,
        label_noise=args.label_noise,
        activation_spike=args.activation_spike,
    )
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown_summary(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
