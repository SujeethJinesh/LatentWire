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
    "prune_uniform_quant",
    "global_activation_protect",
    "verifier_saliency_protect",
    "quant_error_protect",
    "activation_x_verifier_protect",
    "random_protect",
    "oracle_protect",
)


@dataclass(frozen=True)
class ToyProtectedFrontierSelectionConfig:
    seed: int = 0
    calibration_examples: int = 160
    test_examples: int = 192
    atoms: int = 28
    dim: int = 18
    classes: int = 4
    signal_atoms: int = 8
    distractor_atoms: int = 8
    protected_atoms: int = 6
    keep_fraction: float = 0.68
    low_bits: int = 2
    high_bits: int = 8
    signal_scale: float = 2.3
    distractor_scale: float = 7.0
    activation_noise: float = 0.035
    verifier_noise: float = 0.10
    calibration_noise: float = 0.025
    cost_jitter: float = 0.14


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _bytes_for_values(count: int, bits: int) -> int:
    return int(math.ceil(count * bits / 8.0))


def _estimate_uniform_bytes(atoms: int, dim: int, bits: int) -> int:
    return _bytes_for_values(atoms * dim, bits) + 4


def _estimate_mixed_bytes(atoms: int, dim: int, low_bits: int, high_bits: int, protected_atoms: int) -> int:
    protected = max(0, min(int(protected_atoms), int(atoms)))
    low = max(0, int(atoms) - protected)
    index_bytes_per_atom = max(1, math.ceil(math.log2(max(atoms, 2)) / 8.0))
    return (
        _bytes_for_values(low * dim, low_bits)
        + _bytes_for_values(protected * dim, high_bits)
        + 4
        + protected * index_bytes_per_atom
    )


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


def _make_mask(indices: torch.Tensor, atoms: int) -> torch.Tensor:
    mask = torch.zeros(atoms, dtype=torch.bool, device=indices.device)
    if indices.numel() > 0:
        mask[indices] = True
    return mask


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _make_problem(config: ToyProtectedFrontierSelectionConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    if config.classes > config.dim:
        raise ValueError("classes must be <= dim")
    if config.signal_atoms + config.distractor_atoms > config.atoms:
        raise ValueError("signal_atoms + distractor_atoms must be <= atoms")

    raw_prototypes = torch.randn(config.classes, config.dim, generator=gen, dtype=torch.float32)
    prototypes = _normalize_rows(raw_prototypes)
    classifier = prototypes.T.contiguous()

    atom_values = torch.randn(config.atoms, config.dim, generator=gen, dtype=torch.float32) * 0.35
    utility = torch.zeros(config.atoms, dtype=torch.float32)
    atom_class = torch.full((config.atoms,), -1, dtype=torch.long)

    for atom in range(config.signal_atoms):
        cls = atom % config.classes
        direction = prototypes[cls] + 0.18 * torch.randn(config.dim, generator=gen, dtype=torch.float32)
        atom_values[atom] = float(config.signal_scale) * direction
        utility[atom] = 1.4 + 0.18 * (config.signal_atoms - atom)
        atom_class[atom] = cls

    start = config.signal_atoms
    end = start + config.distractor_atoms
    for atom in range(start, end):
        raw = torch.randn(config.dim, generator=gen, dtype=torch.float32)
        projection = prototypes.T @ (prototypes @ raw)
        orthogonal = raw - projection
        atom_values[atom] = float(config.distractor_scale) * orthogonal / orthogonal.norm().clamp_min(1e-8)
        utility[atom] = -0.65 - 0.05 * (atom - start)

    if end < config.atoms:
        atom_values[end:] = 0.45 * torch.randn(config.atoms - end, config.dim, generator=gen, dtype=torch.float32)
        utility[end:] = 0.05 * torch.randn(config.atoms - end, generator=gen, dtype=torch.float32)

    atom_cost = 1.0 + 0.20 * torch.arange(config.atoms, dtype=torch.float32) / max(float(config.atoms - 1), 1.0)
    atom_cost = atom_cost + float(config.cost_jitter) * torch.rand(config.atoms, generator=gen)

    return {
        "atom_values": atom_values.float(),
        "classifier": classifier.float(),
        "utility": utility.float(),
        "atom_cost": atom_cost.float(),
        "atom_class": atom_class,
        "prototypes": prototypes.float(),
    }


def _make_examples(
    config: ToyProtectedFrontierSelectionConfig,
    *,
    count: int,
    seed_offset: int,
    problem: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    gen = _make_generator(config.seed + seed_offset)
    examples: list[dict[str, torch.Tensor]] = []
    atom_class = problem["atom_class"]
    utility = problem["utility"]

    for _ in range(count):
        label = int(torch.randint(0, config.classes, (1,), generator=gen).item())
        activations = 0.018 + float(config.activation_noise) * torch.rand(config.atoms, generator=gen)

        signal_match = atom_class.eq(label)
        signal_other = atom_class.ge(0) & ~signal_match
        activations[signal_match] += 1.65 + 0.30 * torch.rand(int(signal_match.sum().item()), generator=gen)
        activations[signal_other] += 0.12 * torch.rand(int(signal_other.sum().item()), generator=gen)

        distractor_start = config.signal_atoms
        distractor_end = config.signal_atoms + config.distractor_atoms
        if distractor_start < distractor_end:
            activations[distractor_start:distractor_end] += 1.25 + 0.45 * torch.rand(config.distractor_atoms, generator=gen)

        activations = activations + float(config.calibration_noise) * torch.randn(config.atoms, generator=gen)
        activations = activations.clamp_min(0.0)
        activations = activations / activations.sum().clamp_min(1e-8)

        target_summary = torch.einsum("a,ad->d", activations, problem["atom_values"])
        target_summary = target_summary / activations.sum().clamp_min(1e-8)
        target_label = torch.argmax(target_summary @ problem["classifier"], dim=-1)
        verifier_score = activations * utility
        examples.append(
            {
                "activations": activations.float(),
                "target_summary": target_summary.float(),
                "target_label": target_label.long(),
                "verifier_score": verifier_score.float(),
            }
        )
    return examples


def _build_dataset(
    config: ToyProtectedFrontierSelectionConfig,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    problem = _make_problem(config)
    calibration = _make_examples(config, count=config.calibration_examples, seed_offset=11_000, problem=problem)
    test = _make_examples(config, count=config.test_examples, seed_offset=21_000, problem=problem)
    return problem, calibration, test


def _stack(field: str, examples: Sequence[dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([example[field] for example in examples], dim=0)


def _select_keep_mask(scores: torch.Tensor, atom_cost: torch.Tensor, *, keep_fraction: float) -> torch.Tensor:
    total_cost = float(atom_cost.sum().item())
    budget = float(keep_fraction) * total_cost
    order = torch.argsort(scores / atom_cost.clamp_min(1e-8), descending=True)
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


def _select_protected_mask(scores: torch.Tensor, keep_mask: torch.Tensor, protected_atoms: int) -> torch.Tensor:
    masked = scores.clone()
    masked[~keep_mask] = float("-inf")
    available = int(keep_mask.sum().item())
    selected = _select_topk(masked, min(int(protected_atoms), available))
    return _make_mask(selected, keep_mask.numel())


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
    kept_low = keep_mask & ~protected_mask
    kept_high = keep_mask & protected_mask
    if kept_low.any():
        recon_values[kept_low] = _symmetric_quantize(atom_values[kept_low], low_bits)
    if kept_high.any():
        recon_values[kept_high] = _symmetric_quantize(atom_values[kept_high], high_bits)
    recon_values[~keep_mask] = 0.0
    kept_activations = example["activations"] * keep_mask.float()
    return torch.einsum("a,ad->d", kept_activations, recon_values) / kept_activations.sum().clamp_min(1e-8)


def _mask_overlap(selected: torch.Tensor, oracle: torch.Tensor) -> float:
    selected_set = set(int(i) for i in torch.where(selected)[0].tolist())
    oracle_set = set(int(i) for i in torch.where(oracle)[0].tolist())
    if not oracle_set:
        return 1.0 if not selected_set else 0.0
    return len(selected_set & oracle_set) / float(len(oracle_set))


def _row_metrics(
    *,
    method: str,
    config: ToyProtectedFrontierSelectionConfig,
    baseline_correct: torch.Tensor,
    recon: torch.Tensor,
    example: dict[str, torch.Tensor],
    problem: dict[str, torch.Tensor],
    keep_mask: torch.Tensor,
    protected_mask: torch.Tensor,
    oracle_keep: torch.Tensor,
    oracle_protected: torch.Tensor,
    bytes_proxy: int,
    compute_proxy: float,
) -> dict[str, Any]:
    pred = torch.argmax(recon @ problem["classifier"], dim=-1)
    method_correct = pred.eq(example["target_label"])
    helpful_mask = problem["utility"] > 0.25
    pruned_mask = ~keep_mask
    protected_helpful = float((protected_mask & helpful_mask).float().sum().item())
    protected_total = float(protected_mask.float().sum().item())
    pruned_helpful = float((pruned_mask & helpful_mask).float().sum().item())
    helpful_total = float(helpful_mask.float().sum().item())
    pruned_total = float(pruned_mask.float().sum().item())

    return {
        "method": method,
        "seed": int(config.seed),
        "accuracy": float(method_correct.float().item()),
        "mse": float(F.mse_loss(recon, example["target_summary"]).item()),
        "prune_rate": float(pruned_mask.float().mean().item()),
        "kept_rate": float(keep_mask.float().mean().item()),
        "protected_rate": float(protected_mask.float().mean().item()),
        "kept_atoms": int(keep_mask.sum().item()),
        "protected_atoms": int(protected_mask.sum().item()),
        "bytes_proxy": float(bytes_proxy),
        "compute_proxy": float(compute_proxy),
        "missed_help_rate": pruned_helpful / max(helpful_total, 1.0),
        "false_prune_rate": pruned_helpful / max(pruned_total, 1.0),
        "top_atom_preservation_rate": float(_mask_overlap(keep_mask, oracle_keep)),
        "protected_oracle_preservation_rate": float(_mask_overlap(protected_mask, oracle_protected)),
        "protection_precision_rate": protected_helpful / max(protected_total, 1.0),
        "help_vs_prune_uniform_quant": float((method_correct & ~baseline_correct).float().item()),
        "harm_vs_prune_uniform_quant": float((~method_correct & baseline_correct).float().item()),
    }


def _mean_rows(rows: Sequence[dict[str, Any]], *, baseline_accuracy: float, baseline_mse: float) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty row set")
    method = str(rows[0]["method"])
    numeric_keys = [
        "accuracy",
        "mse",
        "prune_rate",
        "kept_rate",
        "protected_rate",
        "kept_atoms",
        "protected_atoms",
        "bytes_proxy",
        "compute_proxy",
        "missed_help_rate",
        "false_prune_rate",
        "top_atom_preservation_rate",
        "protected_oracle_preservation_rate",
        "protection_precision_rate",
        "help_vs_prune_uniform_quant",
        "harm_vs_prune_uniform_quant",
    ]
    summary = {key: float(sum(float(row[key]) for row in rows) / len(rows)) for key in numeric_keys}
    summary["method"] = method
    summary["accuracy_delta_vs_prune_uniform_quant"] = float(summary["accuracy"] - baseline_accuracy)
    summary["mse_delta_vs_prune_uniform_quant"] = float(summary["mse"] - baseline_mse)
    return summary


def run_experiment(config: ToyProtectedFrontierSelectionConfig) -> dict[str, Any]:
    problem, calibration, test = _build_dataset(config)
    atom_values = problem["atom_values"]
    low_quant_error = (atom_values - _symmetric_quantize(atom_values, config.low_bits)).pow(2).sum(dim=-1).sqrt()
    calibration_activations = _stack("activations", calibration)
    calibration_energy = calibration_activations.pow(2).mean(dim=0)
    calibration_saliency = (_stack("verifier_score", calibration).clamp_min(0.0) * calibration_activations).mean(dim=0)

    per_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    baseline_records: list[tuple[torch.Tensor, float]] = []

    for index, example in enumerate(test):
        noise = torch.randn(
            config.atoms,
            generator=_make_generator(config.seed + 31_000 + index),
            dtype=torch.float32,
        )
        verifier_scores = example["verifier_score"] + float(config.verifier_noise) * noise
        keep_scores = verifier_scores.clamp_min(0.0) + 0.10 * calibration_saliency
        keep_mask = _select_keep_mask(keep_scores, problem["atom_cost"], keep_fraction=config.keep_fraction)

        oracle_keep = _select_keep_mask(
            (example["activations"] * problem["utility"].clamp_min(0.0)),
            problem["atom_cost"],
            keep_fraction=config.keep_fraction,
        )
        baseline_protected = torch.zeros(config.atoms, dtype=torch.bool)
        baseline_recon = _predict_summary(
            example,
            atom_values=atom_values,
            keep_mask=keep_mask,
            protected_mask=baseline_protected,
            low_bits=config.low_bits,
            high_bits=config.high_bits,
        )
        baseline_pred = torch.argmax(baseline_recon @ problem["classifier"], dim=-1)
        baseline_correct = baseline_pred.eq(example["target_label"])
        baseline_mse = float(F.mse_loss(baseline_recon, example["target_summary"]).item())
        baseline_records.append((baseline_correct, baseline_mse))

        oracle_protect_scores = torch.full((config.atoms,), float("-inf"), dtype=torch.float32)
        for atom in torch.where(keep_mask)[0].tolist():
            single_protected = baseline_protected.clone()
            single_protected[atom] = True
            single_recon = _predict_summary(
                example,
                atom_values=atom_values,
                keep_mask=keep_mask,
                protected_mask=single_protected,
                low_bits=config.low_bits,
                high_bits=config.high_bits,
            )
            single_mse = float(F.mse_loss(single_recon, example["target_summary"]).item())
            oracle_protect_scores[atom] = baseline_mse - single_mse
        oracle_protected = _select_protected_mask(oracle_protect_scores, keep_mask, config.protected_atoms)

        random_scores = torch.rand(config.atoms, generator=_make_generator(config.seed + 47_000 + index))
        method_scores: dict[str, torch.Tensor] = {
            "prune_uniform_quant": torch.full((config.atoms,), float("-inf")),
            "global_activation_protect": calibration_energy,
            "verifier_saliency_protect": verifier_scores.clamp_min(0.0),
            "quant_error_protect": low_quant_error * example["activations"],
            "activation_x_verifier_protect": low_quant_error
            * (0.35 + calibration_energy.sqrt())
            * verifier_scores.clamp_min(0.0),
            "random_protect": random_scores,
            "oracle_protect": oracle_protect_scores,
        }

        for method in METHODS:
            if method == "prune_uniform_quant":
                protected_mask = baseline_protected
            else:
                protected_mask = _select_protected_mask(method_scores[method], keep_mask, config.protected_atoms)
            recon = _predict_summary(
                example,
                atom_values=atom_values,
                keep_mask=keep_mask,
                protected_mask=protected_mask,
                low_bits=config.low_bits,
                high_bits=config.high_bits,
            )
            bytes_proxy = (
                _estimate_uniform_bytes(int(keep_mask.sum().item()), config.dim, config.low_bits)
                if method == "prune_uniform_quant"
                else _estimate_mixed_bytes(
                    int(keep_mask.sum().item()),
                    config.dim,
                    config.low_bits,
                    config.high_bits,
                    int(protected_mask.sum().item()),
                )
            )
            per_method[method].append(
                _row_metrics(
                    method=method,
                    config=config,
                    baseline_correct=baseline_correct,
                    recon=recon,
                    example=example,
                    problem=problem,
                    keep_mask=keep_mask,
                    protected_mask=protected_mask,
                    oracle_keep=oracle_keep,
                    oracle_protected=oracle_protected,
                    bytes_proxy=bytes_proxy,
                    compute_proxy=float(config.dim * 2.0 * keep_mask.float().sum().item()),
                )
            )

    baseline_accuracy = sum(float(correct.float().item()) for correct, _ in baseline_records) / max(len(baseline_records), 1)
    baseline_mse = sum(mse for _, mse in baseline_records) / max(len(baseline_records), 1)
    summary_rows = [
        _mean_rows(per_method[method], baseline_accuracy=baseline_accuracy, baseline_mse=baseline_mse)
        for method in METHODS
    ]
    return {"config": asdict(config), "methods": list(METHODS), "rows": summary_rows}


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]

    def fmt(value: Any) -> str:
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    lines = [
        "# Toy Protected Frontier Selection",
        "",
        "- All methods share the same verifier-pruned frontier; only the high-precision protected subset changes.",
        "- Protected-oracle preservation measures overlap with atoms whose low-bit quantization error most damages positive utility.",
        "",
        "| Method | Accuracy | Acc delta | MSE | MSE delta | Prune rate | Protected rate | Missed help | False prune | Top-atom preservation | Protected-oracle preservation | Protection precision | Bytes proxy | Compute proxy | Help vs prune-uniform | Harm vs prune-uniform |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {accuracy_delta_vs_prune_uniform_quant} | {mse} | {mse_delta_vs_prune_uniform_quant} | {prune_rate} | {protected_rate} | {missed_help_rate} | {false_prune_rate} | {top_atom_preservation_rate} | {protected_oracle_preservation_rate} | {protection_precision_rate} | {bytes_proxy} | {compute_proxy} | {help_vs_prune_uniform_quant} | {harm_vs_prune_uniform_quant} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                accuracy_delta_vs_prune_uniform_quant=fmt(row["accuracy_delta_vs_prune_uniform_quant"]),
                mse=fmt(row["mse"]),
                mse_delta_vs_prune_uniform_quant=fmt(row["mse_delta_vs_prune_uniform_quant"]),
                prune_rate=fmt(row["prune_rate"]),
                protected_rate=fmt(row["protected_rate"]),
                missed_help_rate=fmt(row["missed_help_rate"]),
                false_prune_rate=fmt(row["false_prune_rate"]),
                top_atom_preservation_rate=fmt(row["top_atom_preservation_rate"]),
                protected_oracle_preservation_rate=fmt(row["protected_oracle_preservation_rate"]),
                protection_precision_rate=fmt(row["protection_precision_rate"]),
                bytes_proxy=fmt(row["bytes_proxy"]),
                compute_proxy=fmt(row["compute_proxy"]),
                help_vs_prune_uniform_quant=fmt(row["help_vs_prune_uniform_quant"]),
                harm_vs_prune_uniform_quant=fmt(row["harm_vs_prune_uniform_quant"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy protected-frontier selection ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-examples", type=int, default=160)
    parser.add_argument("--test-examples", type=int, default=192)
    parser.add_argument("--atoms", type=int, default=28)
    parser.add_argument("--dim", type=int, default=18)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--signal-atoms", type=int, default=8)
    parser.add_argument("--distractor-atoms", type=int, default=8)
    parser.add_argument("--protected-atoms", type=int, default=6)
    parser.add_argument("--keep-fraction", type=float, default=0.68)
    parser.add_argument("--low-bits", type=int, default=2)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--signal-scale", type=float, default=2.3)
    parser.add_argument("--distractor-scale", type=float, default=7.0)
    parser.add_argument("--activation-noise", type=float, default=0.035)
    parser.add_argument("--verifier-noise", type=float, default=0.10)
    parser.add_argument("--calibration-noise", type=float, default=0.025)
    parser.add_argument("--cost-jitter", type=float, default=0.14)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyProtectedFrontierSelectionConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        atoms=args.atoms,
        dim=args.dim,
        classes=args.classes,
        signal_atoms=args.signal_atoms,
        distractor_atoms=args.distractor_atoms,
        protected_atoms=args.protected_atoms,
        keep_fraction=args.keep_fraction,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
        signal_scale=args.signal_scale,
        distractor_scale=args.distractor_scale,
        activation_noise=args.activation_noise,
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
