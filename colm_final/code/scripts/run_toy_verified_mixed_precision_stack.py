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


METHODS: tuple[str, ...] = (
    "full_precision",
    "uniform_low_bit",
    "activation_aware_quant_only",
    "verifier_prune_only",
    "prune_then_uniform_quant",
    "prune_then_activation_aware_quant",
    "oracle_stack",
)


@dataclass(frozen=True)
class ToyVerifiedMixedPrecisionStackConfig:
    seed: int = 0
    calibration_examples: int = 160
    test_examples: int = 192
    atoms: int = 24
    dim: int = 16
    signal_atoms: int = 6
    harmful_atoms: int = 6
    protected_atoms: int = 6
    keep_fraction: float = 0.80
    low_bits: int = 3
    high_bits: int = 8
    signal_scale: float = 3.40
    harmful_scale: float = 8.50
    activation_spike: float = 1.35
    verifier_noise: float = 0.05
    calibration_noise: float = 0.03
    cost_jitter: float = 0.18


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _bytes_for_values(count: int, bits: int) -> int:
    return int(math.ceil(count * bits / 8.0))


def _estimate_uniform_bytes(atoms: int, dim: int, bits: int) -> int:
    return _bytes_for_values(atoms * dim, bits) + 4


def _estimate_mixed_bytes(atoms: int, dim: int, low_bits: int, high_bits: int, protected_atoms: int) -> int:
    protected = max(0, min(int(protected_atoms), int(atoms)))
    low = atoms - protected
    index_bytes_per_atom = max(1, math.ceil(math.log2(max(atoms, 2)) / 8.0))
    return _bytes_for_values(low * dim, low_bits) + _bytes_for_values(protected * dim, high_bits) + 4 + protected * index_bytes_per_atom


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


def _make_problem(config: ToyVerifiedMixedPrecisionStackConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    atom_keys = torch.randn(config.atoms, config.dim, generator=gen, dtype=torch.float32)
    atom_keys = atom_keys / atom_keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    atom_values = torch.randn(config.atoms, config.dim, generator=gen, dtype=torch.float32)
    classifier = torch.randn(config.dim, 5, generator=gen, dtype=torch.float32) / math.sqrt(config.dim)

    signal_atoms = max(1, min(int(config.signal_atoms), int(config.atoms)))
    harmful_atoms = max(1, min(int(config.harmful_atoms), max(int(config.atoms) - signal_atoms, 1)))
    neutral_atoms = max(0, int(config.atoms) - signal_atoms - harmful_atoms)
    utility = torch.cat(
        [
            torch.linspace(float(config.signal_scale), 0.55, steps=signal_atoms),
            -torch.linspace(float(config.harmful_scale), 0.35, steps=harmful_atoms),
            0.15 * torch.randn(neutral_atoms, generator=gen, dtype=torch.float32),
        ],
        dim=0,
    )
    utility = utility[torch.randperm(config.atoms, generator=gen)]

    atom_cost = 1.0 + 0.18 * torch.arange(config.atoms, dtype=torch.float32) / max(float(config.atoms - 1), 1.0)
    atom_cost = atom_cost + config.cost_jitter * torch.rand(config.atoms, generator=gen)

    return {
        "atom_keys": atom_keys,
        "atom_values": atom_values,
        "classifier": classifier,
        "utility": utility.float(),
        "atom_cost": atom_cost.float(),
    }


def _make_examples(
    config: ToyVerifiedMixedPrecisionStackConfig,
    *,
    count: int,
    seed_offset: int,
    problem: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    gen = _make_generator(config.seed + seed_offset)
    atom_keys = problem["atom_keys"]
    utility = problem["utility"]
    atom_cost = problem["atom_cost"]

    examples: list[dict[str, torch.Tensor]] = []
    for _ in range(count):
        query = torch.randn(config.dim, generator=gen, dtype=torch.float32)
        key_scores = (atom_keys @ query) / math.sqrt(config.dim)
        activations = torch.softmax(key_scores / 0.82, dim=-1)
        activations = activations + config.calibration_noise * torch.randn(config.atoms, generator=gen, dtype=torch.float32)
        activations = activations.clamp_min(0.0)
        activations = activations / activations.sum().clamp_min(1e-8)
        clean_contrib = activations * utility.clamp_min(0.0)
        target_summary = torch.einsum("a,ad->d", clean_contrib, problem["atom_values"])
        target_summary = target_summary / clean_contrib.sum().clamp_min(1e-8)
        target_summary = target_summary + 0.03 * torch.randn(config.dim, generator=gen, dtype=torch.float32)
        target_label = torch.argmax(target_summary @ problem["classifier"], dim=-1)
        examples.append(
            {
                "query": query.float(),
                "activations": activations.float(),
                "utility": utility.float(),
                "atom_cost": atom_cost.float(),
                "target_summary": target_summary.float(),
                "target_label": target_label.long(),
            }
        )
    return examples


def _build_dataset(config: ToyVerifiedMixedPrecisionStackConfig) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    problem = _make_problem(config)
    calibration = _make_examples(config, count=config.calibration_examples, seed_offset=11_000, problem=problem)
    test = _make_examples(config, count=config.test_examples, seed_offset=21_000, problem=problem)
    return problem, calibration, test


def _stack(field: str, examples: Sequence[dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([example[field] for example in examples], dim=0)


def _make_verifier_scores(example: dict[str, torch.Tensor], *, seed: int, verifier_noise: float) -> torch.Tensor:
    gen = _make_generator(seed)
    utility = example["utility"]
    activations = example["activations"]
    clean_contrib = activations * utility
    step_like_bonus = activations.pow(0.75) * utility.clamp_min(0.0)
    noise = torch.randn(clean_contrib.shape, generator=gen, dtype=clean_contrib.dtype, device=clean_contrib.device)
    return clean_contrib + 0.20 * step_like_bonus + verifier_noise * noise


def _select_keep_mask(scores: torch.Tensor, atom_cost: torch.Tensor, *, keep_fraction: float, frontier: bool) -> torch.Tensor:
    total_cost = float(atom_cost.sum().item())
    budget = float(keep_fraction) * total_cost
    if frontier:
        order = torch.argsort(scores / atom_cost.clamp_min(1e-8), descending=True)
    else:
        order = torch.argsort(scores, descending=True)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    spent = 0.0
    for idx in order.tolist():
        next_spent = spent + float(atom_cost[idx].item())
        if next_spent <= budget or not bool(keep.any()):
            keep[idx] = True
            spent = next_spent
        if spent >= budget:
            break
    if not bool(keep.any()):
        keep[order[0]] = True
    return keep


def _predict_summary(
    example: dict[str, torch.Tensor],
    *,
    atom_values: torch.Tensor,
    keep_mask: torch.Tensor,
    protected_mask: torch.Tensor,
    low_bits: int,
    high_bits: int,
) -> torch.Tensor:
    recon_values = atom_values.clone()
    if keep_mask.any():
        kept_idx = torch.where(keep_mask)[0]
        if protected_mask.any():
            protected_idx = torch.where(protected_mask)[0]
            low_idx = torch.where(keep_mask & ~protected_mask)[0]
            if low_idx.numel() > 0:
                recon_values[low_idx] = _symmetric_quantize(atom_values[low_idx], low_bits)
            if protected_idx.numel() > 0:
                recon_values[protected_idx] = _symmetric_quantize(atom_values[protected_idx], high_bits)
        else:
            recon_values[kept_idx] = _symmetric_quantize(atom_values[kept_idx], low_bits)
    recon_values[~keep_mask] = 0.0
    kept_activations = example["activations"] * keep_mask.float()
    denom = kept_activations.sum().clamp_min(1e-8)
    return torch.einsum("a,ad->d", kept_activations, recon_values) / denom


def _top_atom_preservation(selected: torch.Tensor, oracle: torch.Tensor) -> float:
    selected_set = set(int(i) for i in torch.where(selected)[0].tolist())
    oracle_set = set(int(i) for i in torch.where(oracle)[0].tolist())
    if not oracle_set:
        return 1.0 if not selected_set else 0.0
    return len(selected_set & oracle_set) / float(len(oracle_set))


def _row_metrics(
    *,
    method: str,
    config: ToyVerifiedMixedPrecisionStackConfig,
    full_pred: torch.Tensor,
    recon: torch.Tensor,
    test_example: dict[str, torch.Tensor],
    atom_values: torch.Tensor,
    keep_mask: torch.Tensor,
    protected_mask: torch.Tensor,
    oracle_keep: torch.Tensor,
    oracle_protected: torch.Tensor,
    bytes_proxy: int,
    compute_proxy: float,
) -> dict[str, Any]:
    preds = torch.argmax(recon @ test_example["classifier"], dim=-1)
    full_correct = full_pred.eq(test_example["target_label"])
    method_correct = preds.eq(test_example["target_label"])
    help_vs_full = float((method_correct & ~full_correct).float().mean().item())
    harm_vs_full = float((~method_correct & full_correct).float().mean().item())

    pruned_mask = ~keep_mask
    helpful_mask = test_example["utility"] > 0
    pruned_help = float((pruned_mask & helpful_mask).float().sum().item() / max(helpful_mask.float().sum().item(), 1.0))
    false_prune = float((pruned_mask & helpful_mask).float().sum().item() / max(pruned_mask.float().sum().item(), 1.0))
    prune_rate = float(pruned_mask.float().mean().item())
    kept_rate = float(keep_mask.float().mean().item())
    protected_rate = float(protected_mask.float().mean().item())
    kept_count = int(keep_mask.sum().item())
    protected_count = int(protected_mask.sum().item())

    return {
        "method": method,
        "seed": int(config.seed),
        "atoms": int(config.atoms),
        "dim": int(config.dim),
        "accuracy": float(method_correct.float().mean().item()),
        "mse": float(F.mse_loss(recon, test_example["target_summary"]).item()),
        "prune_rate": prune_rate,
        "kept_rate": kept_rate,
        "protected_rate": protected_rate,
        "kept_atoms": kept_count,
        "protected_atoms": protected_count,
        "bytes_proxy": float(bytes_proxy),
        "compute_proxy": float(compute_proxy),
        "missed_help_rate": pruned_help,
        "false_prune_rate": false_prune,
        "top_atom_preservation_rate": float(_top_atom_preservation(keep_mask, oracle_keep)),
        "protected_atom_preservation_rate": float(_top_atom_preservation(protected_mask, oracle_protected)),
        "help_vs_full_precision": help_vs_full,
        "harm_vs_full_precision": harm_vs_full,
        "selected_atom_indices": [int(i) for i in torch.where(keep_mask)[0].tolist()],
        "protected_atom_indices": [int(i) for i in torch.where(protected_mask)[0].tolist()],
        "oracle_keep_indices": [int(i) for i in torch.where(oracle_keep)[0].tolist()],
        "oracle_protected_indices": [int(i) for i in torch.where(oracle_protected)[0].tolist()],
    }


def run_experiment(config: ToyVerifiedMixedPrecisionStackConfig) -> dict[str, Any]:
    problem, calibration, test = _build_dataset(config)
    atom_values = problem["atom_values"]
    calibration_x = _stack("activations", calibration)
    calibration_energy = calibration_x.pow(2).mean(dim=0)
    calibration_peak = calibration_x.abs().amax(dim=0)
    oracle_protect_score = calibration_energy * problem["utility"].clamp_min(0.0)

    rows: list[dict[str, Any]] = []
    full_preds = []
    for example in test:
        summary = torch.einsum("a,ad->d", example["activations"] * example["utility"].clamp_min(0.0), atom_values)
        summary = summary / (example["activations"] * example["utility"].clamp_min(0.0)).sum().clamp_min(1e-8)
        full_preds.append(torch.argmax(summary @ problem["classifier"], dim=-1))
    full_preds = torch.stack(full_preds, dim=0)

    per_method = {method: [] for method in METHODS}

    for index, example in enumerate(test):
        verifier_scores = _make_verifier_scores(example, seed=config.seed + 31_000 + index, verifier_noise=config.verifier_noise)
        oracle_keep_scores = example["activations"] * example["utility"].clamp_min(0.0)

        keep_full = torch.ones(config.atoms, dtype=torch.bool)
        protected_full = torch.ones(config.atoms, dtype=torch.bool)

        uniform_keep = torch.ones(config.atoms, dtype=torch.bool)
        uniform_protected = torch.zeros(config.atoms, dtype=torch.bool)

        activation_protected = _select_topk(calibration_energy, config.protected_atoms)
        activation_keep = torch.ones(config.atoms, dtype=torch.bool)
        activation_mask = torch.zeros(config.atoms, dtype=torch.bool)
        activation_mask[activation_protected] = True

        verifier_keep = _select_keep_mask(verifier_scores, example["atom_cost"], keep_fraction=config.keep_fraction, frontier=True)
        verifier_protected = verifier_keep.clone()

        prune_uniform_keep = verifier_keep.clone()
        prune_uniform_protected = torch.zeros(config.atoms, dtype=torch.bool)

        prune_activation_keep = verifier_keep.clone()
        keep_energy = calibration_energy * verifier_scores.clamp_min(0.0)
        keep_energy[~prune_activation_keep] = float("-inf")
        protected_from_keep = _select_topk(keep_energy, min(int(config.protected_atoms), int(prune_activation_keep.sum().item())))
        prune_activation_protected = torch.zeros(config.atoms, dtype=torch.bool)
        prune_activation_protected[protected_from_keep] = True

        oracle_keep = _select_keep_mask(oracle_keep_scores, example["atom_cost"], keep_fraction=config.keep_fraction, frontier=True)
        oracle_keep_energy = oracle_protect_score.clone()
        oracle_keep_energy[~oracle_keep] = float("-inf")
        oracle_protected_idx = _select_topk(oracle_keep_energy, min(int(config.protected_atoms), int(oracle_keep.sum().item())))
        oracle_protected = torch.zeros(config.atoms, dtype=torch.bool)
        oracle_protected[oracle_protected_idx] = True

        method_specs = [
            ("full_precision", keep_full, protected_full, _estimate_uniform_bytes(config.atoms, config.dim, 16), float(config.atoms), atom_values.clone()),
            ("uniform_low_bit", uniform_keep, uniform_protected, _estimate_uniform_bytes(config.atoms, config.dim, config.low_bits), float(config.atoms), _symmetric_quantize(atom_values, config.low_bits)),
            (
                "activation_aware_quant_only",
                activation_keep,
                activation_mask,
                _estimate_mixed_bytes(config.atoms, config.dim, config.low_bits, config.high_bits, int(activation_mask.sum().item())),
                float(config.atoms),
                None,
            ),
            (
                "verifier_prune_only",
                verifier_keep,
                verifier_protected,
                _estimate_uniform_bytes(int(verifier_keep.sum().item()), config.dim, 16),
                float(example["atom_cost"][verifier_keep].sum().item()),
                None,
            ),
            (
                "prune_then_uniform_quant",
                prune_uniform_keep,
                prune_uniform_protected,
                _estimate_uniform_bytes(int(prune_uniform_keep.sum().item()), config.dim, config.low_bits),
                float(example["atom_cost"][prune_uniform_keep].sum().item()),
                None,
            ),
            (
                "prune_then_activation_aware_quant",
                prune_activation_keep,
                prune_activation_protected,
                _estimate_mixed_bytes(
                    int(prune_activation_keep.sum().item()),
                    config.dim,
                    config.low_bits,
                    config.high_bits,
                    int(prune_activation_protected.sum().item()),
                ),
                float(example["atom_cost"][prune_activation_keep].sum().item()),
                None,
            ),
            (
                "oracle_stack",
                oracle_keep,
                oracle_protected,
                _estimate_mixed_bytes(
                    int(oracle_keep.sum().item()),
                    config.dim,
                    config.low_bits,
                    config.high_bits,
                    int(oracle_protected.sum().item()),
                ),
                float(example["atom_cost"][oracle_keep].sum().item()),
                None,
            ),
        ]

        for method, keep_mask, protected_mask, bytes_proxy, compute_cost, precomputed_recon in method_specs:
            if precomputed_recon is None:
                recon = _predict_summary(
                    example,
                    atom_values=atom_values,
                    keep_mask=keep_mask,
                    protected_mask=protected_mask,
                    low_bits=config.low_bits,
                    high_bits=config.high_bits,
                )
            else:
                recon = precomputed_recon
                recon = torch.einsum("a,ad->d", example["activations"] * keep_mask.float(), recon) / (example["activations"] * keep_mask.float()).sum().clamp_min(1e-8)

            rows = per_method[method]
            rows.append(
                _row_metrics(
                    method=method,
                    config=config,
                    full_pred=full_preds[index],
                    recon=recon,
                    test_example={
                        **example,
                        "classifier": problem["classifier"],
                    },
                    atom_values=atom_values,
                    keep_mask=keep_mask,
                    protected_mask=protected_mask,
                    oracle_keep=oracle_keep,
                    oracle_protected=oracle_protected,
                    bytes_proxy=bytes_proxy,
                    compute_proxy=float(compute_cost * config.dim * 2.0),
                )
            )

    summary_rows: list[dict[str, Any]] = []
    full_accuracy = sum(row["accuracy"] for row in per_method["full_precision"]) / max(len(per_method["full_precision"]), 1)
    full_mse = sum(row["mse"] for row in per_method["full_precision"]) / max(len(per_method["full_precision"]), 1)
    for method in METHODS:
        subset = per_method[method]
        accuracy = sum(row["accuracy"] for row in subset) / max(len(subset), 1)
        mse = sum(row["mse"] for row in subset) / max(len(subset), 1)
        prune_rate = sum(row["prune_rate"] for row in subset) / max(len(subset), 1)
        kept_rate = sum(row["kept_rate"] for row in subset) / max(len(subset), 1)
        protected_rate = sum(row["protected_rate"] for row in subset) / max(len(subset), 1)
        kept_atoms = sum(row["kept_atoms"] for row in subset) / max(len(subset), 1)
        protected_atoms = sum(row["protected_atoms"] for row in subset) / max(len(subset), 1)
        missed_help = sum(row["missed_help_rate"] for row in subset) / max(len(subset), 1)
        false_prune = sum(row["false_prune_rate"] for row in subset) / max(len(subset), 1)
        top_atom = sum(row["top_atom_preservation_rate"] for row in subset) / max(len(subset), 1)
        protected_top_atom = sum(row["protected_atom_preservation_rate"] for row in subset) / max(len(subset), 1)
        bytes_proxy = sum(row["bytes_proxy"] for row in subset) / max(len(subset), 1)
        compute_proxy = sum(row["compute_proxy"] for row in subset) / max(len(subset), 1)
        summary_rows.append(
            {
                "method": method,
                "accuracy": float(accuracy),
                "mse": float(mse),
                "accuracy_delta_vs_full_precision": float(accuracy - full_accuracy),
                "mse_delta_vs_full_precision": float(mse - full_mse),
                "prune_rate": float(prune_rate),
                "kept_rate": float(kept_rate),
                "protected_rate": float(protected_rate),
                "kept_atoms": float(kept_atoms),
                "protected_atoms": float(protected_atoms),
                "missed_help_rate": float(missed_help),
                "false_prune_rate": float(false_prune),
                "top_atom_preservation_rate": float(top_atom),
                "protected_atom_preservation_rate": float(protected_top_atom),
                "bytes_proxy": float(bytes_proxy),
                "compute_proxy": float(compute_proxy),
                "help_vs_full_precision": float(max(0.0, accuracy - full_accuracy)),
                "harm_vs_full_precision": float(max(0.0, full_accuracy - accuracy)),
            }
        )

    return {"config": asdict(config), "methods": list(METHODS), "rows": summary_rows}


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    lines = [
        "# Toy Verified Mixed-Precision Stack",
        "",
        "- Protected rate = fraction of atoms assigned high precision or left unpruned in a prune-only policy.",
        "- Top-atom preservation = overlap with the oracle selection for the relevant decision stage.",
        "",
        "| Method | Accuracy | MSE | Prune rate | Kept rate | Protected rate | Missed help | False prune | Top-atom preservation | Protected-top preservation | Bytes proxy | Compute proxy | Help vs full | Harm vs full |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {mse} | {prune_rate} | {kept_rate} | {protected_rate} | {missed_help_rate} | {false_prune_rate} | {top_atom_preservation_rate} | {protected_atom_preservation_rate} | {bytes_proxy} | {compute_proxy} | {help_vs_full_precision} | {harm_vs_full_precision} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                mse=fmt(row["mse"]),
                prune_rate=fmt(row["prune_rate"]),
                kept_rate=fmt(row["kept_rate"]),
                protected_rate=fmt(row["protected_rate"]),
                missed_help_rate=fmt(row["missed_help_rate"]),
                false_prune_rate=fmt(row["false_prune_rate"]),
                top_atom_preservation_rate=fmt(row["top_atom_preservation_rate"]),
                protected_atom_preservation_rate=fmt(row["protected_atom_preservation_rate"]),
                bytes_proxy=fmt(row["bytes_proxy"]),
                compute_proxy=fmt(row["compute_proxy"]),
                help_vs_full_precision=fmt(row["help_vs_full_precision"]),
                harm_vs_full_precision=fmt(row["harm_vs_full_precision"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy verifier-guided pruning + activation-aware mixed precision stack.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-examples", type=int, default=160)
    parser.add_argument("--test-examples", type=int, default=192)
    parser.add_argument("--atoms", type=int, default=24)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--signal-atoms", type=int, default=6)
    parser.add_argument("--harmful-atoms", type=int, default=6)
    parser.add_argument("--protected-atoms", type=int, default=6)
    parser.add_argument("--keep-fraction", type=float, default=0.80)
    parser.add_argument("--low-bits", type=int, default=3)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--signal-scale", type=float, default=3.40)
    parser.add_argument("--harmful-scale", type=float, default=8.50)
    parser.add_argument("--activation-spike", type=float, default=1.35)
    parser.add_argument("--verifier-noise", type=float, default=0.05)
    parser.add_argument("--calibration-noise", type=float, default=0.03)
    parser.add_argument("--cost-jitter", type=float, default=0.18)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyVerifiedMixedPrecisionStackConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        atoms=args.atoms,
        dim=args.dim,
        signal_atoms=args.signal_atoms,
        harmful_atoms=args.harmful_atoms,
        protected_atoms=args.protected_atoms,
        keep_fraction=args.keep_fraction,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
        signal_scale=args.signal_scale,
        harmful_scale=args.harmful_scale,
        activation_spike=args.activation_spike,
        verifier_noise=args.verifier_noise,
        calibration_noise=args.calibration_noise,
        cost_jitter=args.cost_jitter,
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
