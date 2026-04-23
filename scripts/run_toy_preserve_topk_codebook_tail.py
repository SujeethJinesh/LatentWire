#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_toy_activation_aware_atom_quant as base_quant


@dataclass(frozen=True)
class ToyPreserveTopKCodebookTailConfig:
    seed: int = 0
    calibration_samples: int = 160
    test_samples: int = 192
    atoms: int = 32
    signal_atoms: int = 6
    outlier_atoms: int = 4
    preserved_atoms: int = 8
    low_bits: int = 3
    high_bits: int = 8
    codebook_size: int = 12
    codebook_iters: int = 10
    residual_lambda: float = 1e-3
    signal_scale: float = 3.25
    outlier_scale: float = 8.5
    label_noise: float = 0.18
    activation_spike: float = 1.4


def _base_config(config: ToyPreserveTopKCodebookTailConfig) -> base_quant.ToyActivationAwareAtomQuantConfig:
    return base_quant.ToyActivationAwareAtomQuantConfig(
        seed=config.seed,
        calibration_samples=config.calibration_samples,
        test_samples=config.test_samples,
        atoms=config.atoms,
        signal_atoms=config.signal_atoms,
        outlier_atoms=config.outlier_atoms,
        protected_atoms=config.preserved_atoms,
        low_bits=config.low_bits,
        high_bits=config.high_bits,
        signal_scale=config.signal_scale,
        outlier_scale=config.outlier_scale,
        label_noise=config.label_noise,
        activation_spike=config.activation_spike,
    )


def _fit_codebook(x: torch.Tensor, size: int, *, seed: int, iters: int) -> torch.Tensor:
    if x.shape[0] == 0 or x.shape[1] == 0:
        return torch.zeros(max(1, size), x.shape[1], dtype=x.dtype)
    size = max(1, min(int(size), int(x.shape[0])))
    gen = base_quant._make_generator(seed)
    indices = torch.randperm(x.shape[0], generator=gen)[:size]
    codebook = x[indices].clone()
    for _ in range(max(1, int(iters))):
        distances = torch.cdist(x, codebook)
        winners = distances.argmin(dim=-1)
        updated = []
        for atom in range(size):
            mask = winners == atom
            if mask.any():
                updated.append(x[mask].mean(dim=0))
            else:
                updated.append(codebook[atom])
        codebook = torch.stack(updated, dim=0)
    return codebook


def _apply_codebook(x: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape[1] == 0:
        return x.clone(), torch.zeros(x.shape[0], dtype=torch.long)
    distances = torch.cdist(x, codebook)
    winners = distances.argmin(dim=-1)
    return codebook[winners], winners


def _fit_ridge(source: torch.Tensor, target: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat([source, torch.ones(source.shape[0], 1, dtype=source.dtype)], dim=1)
    xtx = x.T @ x + float(lam) * torch.eye(x.shape[1], dtype=source.dtype)
    xty = x.T @ target
    weight = torch.linalg.solve(xtx, xty)
    return weight[:-1], weight[-1]


def _predict_ridge(source: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return source @ weight + bias


def _bytes_for_values(count: int, bits: int) -> float:
    return float(math.ceil(count * bits / 8.0))


def _codebook_bytes_per_example(*, preserved_atoms: int, atoms: int, high_bits: int, codebook_size: int, test_samples: int) -> float:
    tail_atoms = max(0, int(atoms) - int(preserved_atoms))
    index_bits = max(1, math.ceil(math.log2(max(int(codebook_size), 2))))
    table_bits = int(codebook_size) * tail_atoms * int(high_bits)
    return _bytes_for_values(preserved_atoms, high_bits) + float(math.ceil(index_bits / 8.0)) + table_bits / max(int(test_samples), 1) / 8.0


def _row(
    *,
    method: str,
    config: ToyPreserveTopKCodebookTailConfig,
    recon: torch.Tensor,
    test_x: torch.Tensor,
    labels: torch.Tensor,
    true_w: torch.Tensor,
    selected_atoms: torch.Tensor,
    oracle_atoms: torch.Tensor,
    bytes_proxy: float,
    bit_budget: float,
    codebook_assignments: torch.Tensor | None = None,
) -> dict[str, Any]:
    preds = (recon @ true_w >= 0).to(torch.long)
    full_preds = (test_x @ true_w >= 0).to(torch.long)
    method_correct = preds.eq(labels)
    full_correct = full_preds.eq(labels)
    selected = set(int(i) for i in selected_atoms.tolist())
    oracle = set(int(i) for i in oracle_atoms.tolist())
    topk_overlap = 1.0 if len(oracle) == 0 else len(selected & oracle) / float(len(oracle))
    row = {
        "method": method,
        "seed": int(config.seed),
        "atoms": int(config.atoms),
        "preserved_atoms": int(selected_atoms.numel()),
        "codebook_size": int(config.codebook_size),
        "low_bits": int(config.low_bits),
        "high_bits": int(config.high_bits),
        "bit_budget": float(bit_budget),
        "bytes_proxy": float(bytes_proxy),
        "accuracy": float(method_correct.float().mean().item()),
        "mse": float(F.mse_loss(recon, test_x).item()),
        "cosine": base_quant._cosine_mean(recon, test_x),
        "help_vs_full_precision": float((method_correct & ~full_correct).float().mean().item()),
        "harm_vs_full_precision": float((~method_correct & full_correct).float().mean().item()),
        "topk_overlap_with_oracle": float(topk_overlap),
        "selected_atoms": [int(i) for i in selected_atoms.tolist()],
        "oracle_atoms": [int(i) for i in oracle_atoms.tolist()],
    }
    if codebook_assignments is not None and codebook_assignments.numel() > 0:
        counts = torch.bincount(codebook_assignments, minlength=config.codebook_size).to(torch.float32)
        probs = counts / counts.sum().clamp_min(1e-8)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
        row["codebook_perplexity"] = float(torch.exp(entropy).item())
    else:
        row["codebook_perplexity"] = None
    return row


def run_experiment(config: ToyPreserveTopKCodebookTailConfig) -> dict[str, Any]:
    base_config = _base_config(config)
    calibration_x, _, _ = base_quant._make_latents(samples=config.calibration_samples, config=base_config, seed_offset=0)
    test_x, labels, true_w = base_quant._make_latents(samples=config.test_samples, config=base_config, seed_offset=9_701)

    calibration_energy = calibration_x.pow(2).mean(dim=0)
    calibration_oracle = calibration_x.std(dim=0, unbiased=False) * torch.abs(true_w)
    selected_atoms = base_quant._select_topk(calibration_energy, config.preserved_atoms)
    oracle_atoms = base_quant._select_topk(calibration_oracle, config.preserved_atoms)
    tail_mask = torch.ones(config.atoms, dtype=torch.bool)
    tail_mask[selected_atoms] = False

    rows: list[dict[str, Any]] = []

    uniform_recon = base_quant._symmetric_quantize(test_x, config.low_bits)
    rows.append(
        _row(
            method="uniform_low_bit",
            config=config,
            recon=uniform_recon,
            test_x=test_x,
            labels=labels,
            true_w=true_w,
            selected_atoms=torch.empty(0, dtype=torch.long),
            oracle_atoms=oracle_atoms,
            bytes_proxy=_bytes_for_values(config.atoms, config.low_bits),
            bit_budget=float(config.low_bits),
        )
    )

    topk_recon = test_x.clone()
    if tail_mask.any():
        topk_recon[:, tail_mask] = base_quant._symmetric_quantize(test_x[:, tail_mask], config.low_bits)
    if selected_atoms.numel() > 0:
        topk_recon[:, selected_atoms] = base_quant._symmetric_quantize(test_x[:, selected_atoms], config.high_bits)
    rows.append(
        _row(
            method="preserve_topk_uniform_tail",
            config=config,
            recon=topk_recon,
            test_x=test_x,
            labels=labels,
            true_w=true_w,
            selected_atoms=selected_atoms,
            oracle_atoms=oracle_atoms,
            bytes_proxy=_bytes_for_values(selected_atoms.numel(), config.high_bits)
            + _bytes_for_values(config.atoms - int(selected_atoms.numel()), config.low_bits),
            bit_budget=float(
                (int(selected_atoms.numel()) * config.high_bits + (config.atoms - int(selected_atoms.numel())) * config.low_bits)
                / float(config.atoms)
            ),
        )
    )

    if tail_mask.any():
        calibration_tail = calibration_x[:, tail_mask]
        test_tail = test_x[:, tail_mask]
        codebook = _fit_codebook(
            calibration_tail,
            config.codebook_size,
            seed=config.seed + 101,
            iters=config.codebook_iters,
        )
        calib_codebook_recon, _ = _apply_codebook(calibration_tail, codebook)
        test_codebook_recon, test_assignments = _apply_codebook(test_tail, codebook)
        codebook_recon = test_x.clone()
        codebook_recon[:, tail_mask] = test_codebook_recon
        if selected_atoms.numel() > 0:
            codebook_recon[:, selected_atoms] = base_quant._symmetric_quantize(test_x[:, selected_atoms], config.high_bits)
        rows.append(
            _row(
                method="preserve_topk_codebook_tail",
                config=config,
                recon=codebook_recon,
                test_x=test_x,
                labels=labels,
                true_w=true_w,
                selected_atoms=selected_atoms,
                oracle_atoms=oracle_atoms,
                bytes_proxy=_codebook_bytes_per_example(
                    preserved_atoms=int(selected_atoms.numel()),
                    atoms=config.atoms,
                    high_bits=config.high_bits,
                    codebook_size=config.codebook_size,
                    test_samples=config.test_samples,
                ),
                bit_budget=float(
                    (int(selected_atoms.numel()) * config.high_bits + math.ceil(math.log2(max(config.codebook_size, 2))))
                    / float(config.atoms)
                ),
                codebook_assignments=test_assignments,
            )
        )

        weight, bias = _fit_ridge(calib_codebook_recon, calibration_tail, config.residual_lambda)
        residual_tail = _predict_ridge(test_codebook_recon, weight, bias)
        residual_recon = test_x.clone()
        residual_recon[:, tail_mask] = residual_tail
        if selected_atoms.numel() > 0:
            residual_recon[:, selected_atoms] = base_quant._symmetric_quantize(test_x[:, selected_atoms], config.high_bits)
        rows.append(
            _row(
                method="preserve_topk_codebook_tail_residual_fix",
                config=config,
                recon=residual_recon,
                test_x=test_x,
                labels=labels,
                true_w=true_w,
                selected_atoms=selected_atoms,
                oracle_atoms=oracle_atoms,
                bytes_proxy=_codebook_bytes_per_example(
                    preserved_atoms=int(selected_atoms.numel()),
                    atoms=config.atoms,
                    high_bits=config.high_bits,
                    codebook_size=config.codebook_size,
                    test_samples=config.test_samples,
                )
                + 8.0,
                bit_budget=float(
                    (int(selected_atoms.numel()) * config.high_bits + math.ceil(math.log2(max(config.codebook_size, 2))) + 8.0)
                    / float(config.atoms)
                ),
                codebook_assignments=test_assignments,
            )
        )

    return {"config": asdict(config), "rows": rows}


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = [
        "# Toy Preserve-TopK Codebook Tail",
        "",
        "| Method | Acc | MSE | Cosine | Bit budget | Bytes proxy | TopK overlap | Codebook ppl | Help vs full | Harm vs full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        ppl = "-" if row["codebook_perplexity"] is None else f'{row["codebook_perplexity"]:.4f}'
        lines.append(
            "| {method} | {accuracy:.4f} | {mse:.4f} | {cosine:.4f} | {bit_budget:.4f} | {bytes_proxy:.2f} | {topk_overlap_with_oracle:.4f} | {ppl} | {help_vs_full_precision:.4f} | {harm_vs_full_precision:.4f} |".format(
                ppl=ppl,
                **row,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy preserve-topk plus codebook-tail bridge ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=160)
    parser.add_argument("--test-samples", type=int, default=192)
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--signal-atoms", type=int, default=6)
    parser.add_argument("--outlier-atoms", type=int, default=4)
    parser.add_argument("--preserved-atoms", type=int, default=8)
    parser.add_argument("--low-bits", type=int, default=3)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=12)
    parser.add_argument("--codebook-iters", type=int, default=10)
    parser.add_argument("--residual-lambda", type=float, default=1e-3)
    parser.add_argument("--signal-scale", type=float, default=3.25)
    parser.add_argument("--outlier-scale", type=float, default=8.5)
    parser.add_argument("--label-noise", type=float, default=0.18)
    parser.add_argument("--activation-spike", type=float, default=1.4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyPreserveTopKCodebookTailConfig(
        seed=args.seed,
        calibration_samples=args.calibration_samples,
        test_samples=args.test_samples,
        atoms=args.atoms,
        signal_atoms=args.signal_atoms,
        outlier_atoms=args.outlier_atoms,
        preserved_atoms=args.preserved_atoms,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
        codebook_size=args.codebook_size,
        codebook_iters=args.codebook_iters,
        residual_lambda=args.residual_lambda,
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
