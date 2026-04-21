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
    "uniform_3_bit",
    "uniform_4_bit",
    "quant_error_target_bpw_allocator",
    "exact_patch_target_bpw_allocator",
    "universal_feature_persistence_allocator",
    "random_allocator",
    "oracle_allocator",
)


@dataclass(frozen=True)
class ToyMixedBitRouteAtomAllocatorConfig:
    seed: int = 0
    calibration_examples: int = 128
    test_examples: int = 160
    atoms: int = 32
    dim: int = 20
    classes: int = 4
    universal_features: int = 18
    signal_atoms: int = 8
    outlier_atoms: int = 10
    low_bits: int = 3
    mid_bits: int = 4
    high_bits: int = 8
    target_bpw: float = 4.0
    signal_scale: float = 2.5
    outlier_scale: float = 9.0
    activation_noise: float = 0.032
    calibration_noise: float = 0.018
    dictionary_noise: float = 0.055


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (int(bits) - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _quantize_by_atom_bits(atom_values: torch.Tensor, bit_allocation: torch.Tensor) -> torch.Tensor:
    recon = torch.empty_like(atom_values)
    for bits in sorted(set(int(bit) for bit in bit_allocation.tolist())):
        mask = bit_allocation.eq(bits)
        if bool(mask.any()):
            recon[mask] = _symmetric_quantize(atom_values[mask], bits)
    return recon


def _bytes_for_values(count: int, bits: int) -> int:
    return int(math.ceil(int(count) * int(bits) / 8.0))


def _bytes_proxy(bit_allocation: torch.Tensor, dim: int) -> int:
    total = 4
    for bits in sorted(set(int(bit) for bit in bit_allocation.tolist())):
        total += _bytes_for_values(int(bit_allocation.eq(bits).sum().item()) * int(dim), bits)
    return total


def _bit_histogram(bit_allocation: torch.Tensor) -> dict[str, int]:
    return {str(bits): int(bit_allocation.eq(bits).sum().item()) for bits in sorted(set(int(b) for b in bit_allocation.tolist()))}


def _target_high_count(config: ToyMixedBitRouteAtomAllocatorConfig) -> int:
    numerator = float(config.target_bpw) - float(config.low_bits)
    denominator = float(config.high_bits) - float(config.low_bits)
    return max(0, min(config.atoms, int(round(config.atoms * numerator / max(denominator, 1e-8)))))


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values.float(), stable=True)
    ranks = torch.empty_like(values, dtype=torch.float32)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and float(sorted_values[end].item()) == float(sorted_values[start].item()):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return 0.0
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float((left_centered @ right_centered / denom).clamp(-1.0, 1.0).item())


def _topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    mask = torch.zeros_like(scores, dtype=torch.bool)
    k = max(0, min(int(k), int(scores.numel())))
    if k:
        mask[torch.topk(scores, k=k, largest=True).indices] = True
    return mask


def _mask_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = int(right.sum().item())
    if denom == 0:
        return 1.0 if int(left.sum().item()) == 0 else 0.0
    return float((left & right).float().sum().item() / denom)


def _make_problem(config: ToyMixedBitRouteAtomAllocatorConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    if config.classes > config.dim:
        raise ValueError("classes must be <= dim")
    if config.signal_atoms + config.outlier_atoms > config.atoms:
        raise ValueError("signal_atoms + outlier_atoms must be <= atoms")
    if config.classes * 2 > config.universal_features:
        raise ValueError("universal_features must be at least 2 * classes")

    prototypes = _normalize_rows(torch.randn(config.classes, config.dim, generator=gen))
    classifier = prototypes.T.contiguous()
    source_dictionary = _normalize_rows(torch.randn(config.universal_features, config.dim, generator=gen))
    target_dictionary = _normalize_rows(
        source_dictionary + 0.10 * torch.randn(config.universal_features, config.dim, generator=gen)
    )

    source_codes = 0.012 * torch.rand(config.atoms, config.universal_features, generator=gen)
    target_codes = 0.012 * torch.rand(config.atoms, config.universal_features, generator=gen)
    atom_class = torch.full((config.atoms,), -1, dtype=torch.long)
    utility = torch.zeros(config.atoms, dtype=torch.float32)

    for atom in range(config.signal_atoms):
        cls = atom % config.classes
        second_feature = config.classes + (atom // config.classes)
        source_codes[atom, cls] = 1.55 + 0.05 * atom
        target_codes[atom, cls] = 1.60 + 0.05 * atom
        source_codes[atom, second_feature] = 0.62
        target_codes[atom, second_feature] = 0.65
        atom_class[atom] = cls
        utility[atom] = 1.6 + 0.08 * (config.signal_atoms - atom)

    outlier_start = config.signal_atoms
    outlier_end = outlier_start + config.outlier_atoms
    for pos, atom in enumerate(range(outlier_start, outlier_end)):
        feature = config.classes * 2 + pos % max(1, config.universal_features - config.classes * 2)
        shifted_feature = config.classes * 2 + (pos * 5 + 2) % max(1, config.universal_features - config.classes * 2)
        source_codes[atom, feature] = 2.0 + 0.10 * pos
        target_codes[atom, shifted_feature] = 2.0 + 0.10 * pos
        utility[atom] = -0.72 - 0.03 * pos

    if outlier_end < config.atoms:
        source_codes[outlier_end:] += 0.12 * torch.rand(config.atoms - outlier_end, config.universal_features, generator=gen)
        target_codes[outlier_end:] += 0.12 * torch.rand(config.atoms - outlier_end, config.universal_features, generator=gen)
        utility[outlier_end:] = 0.03 * torch.randn(config.atoms - outlier_end, generator=gen)

    atom_values = target_codes @ target_dictionary
    for atom in range(config.signal_atoms):
        cls = int(atom_class[atom].item())
        atom_values[atom] = atom_values[atom] + float(config.signal_scale) * prototypes[cls]

    for atom in range(outlier_start, outlier_end):
        raw = atom_values[atom] + float(config.outlier_scale) * torch.randn(config.dim, generator=gen)
        projection = prototypes.T @ (prototypes @ raw)
        atom_values[atom] = raw - projection

    outlier_scores = (atom_values - _symmetric_quantize(atom_values, config.low_bits)).pow(2).mean(dim=-1)
    outlier_mask = _topk_mask(outlier_scores, config.outlier_atoms)

    return {
        "atom_values": atom_values.float(),
        "classifier": classifier.float(),
        "utility": utility.float(),
        "atom_class": atom_class,
        "source_codes": source_codes.float(),
        "target_codes": target_codes.float(),
        "outlier_mask": outlier_mask,
    }


def _make_examples(
    config: ToyMixedBitRouteAtomAllocatorConfig,
    *,
    count: int,
    seed_offset: int,
    problem: dict[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    gen = _make_generator(config.seed + seed_offset)
    atom_class = problem["atom_class"]
    examples: list[dict[str, torch.Tensor]] = []
    for _ in range(count):
        label = int(torch.randint(0, config.classes, (1,), generator=gen).item())
        activations = 0.018 + float(config.activation_noise) * torch.rand(config.atoms, generator=gen)
        signal_match = atom_class.eq(label)
        signal_other = atom_class.ge(0) & ~signal_match
        activations[signal_match] += 1.72 + 0.28 * torch.rand(int(signal_match.sum().item()), generator=gen)
        activations[signal_other] += 0.10 * torch.rand(int(signal_other.sum().item()), generator=gen)
        activations[config.signal_atoms : config.signal_atoms + config.outlier_atoms] += 1.45 + 0.55 * torch.rand(
            config.outlier_atoms, generator=gen
        )
        activations = (activations + float(config.calibration_noise) * torch.randn(config.atoms, generator=gen)).clamp_min(
            0.0
        )
        activations = activations / activations.sum().clamp_min(1e-8)
        target_summary = torch.einsum("a,ad->d", activations, problem["atom_values"])
        target_label = torch.argmax(target_summary @ problem["classifier"])
        examples.append(
            {
                "activations": activations.float(),
                "target_summary": target_summary.float(),
                "target_label": target_label.long(),
            }
        )
    return examples


def _build_dataset(
    config: ToyMixedBitRouteAtomAllocatorConfig,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    problem = _make_problem(config)
    calibration = _make_examples(config, count=config.calibration_examples, seed_offset=11_000, problem=problem)
    test = _make_examples(config, count=config.test_examples, seed_offset=21_000, problem=problem)
    return problem, calibration, test


def _stack(field: str, examples: Sequence[dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([example[field] for example in examples], dim=0)


def _summary(example: dict[str, torch.Tensor], atom_values: torch.Tensor) -> torch.Tensor:
    return torch.einsum("a,ad->d", example["activations"], atom_values)


def _exact_patch_scores(
    examples: Sequence[dict[str, torch.Tensor]],
    *,
    config: ToyMixedBitRouteAtomAllocatorConfig,
    problem: dict[str, torch.Tensor],
) -> torch.Tensor:
    atom_values = problem["atom_values"]
    classifier = problem["classifier"]
    low_bits = torch.full((config.atoms,), config.low_bits, dtype=torch.long)
    low_values = _quantize_by_atom_bits(atom_values, low_bits)
    low_losses = [float(F.mse_loss(_summary(example, low_values), example["target_summary"]).item()) for example in examples]
    scores = torch.zeros(config.atoms, dtype=torch.float32)
    for atom in range(config.atoms):
        patch_bits = low_bits.clone()
        patch_bits[atom] = config.high_bits
        patched_values = _quantize_by_atom_bits(atom_values, patch_bits)
        total = 0.0
        for baseline_loss, example in zip(low_losses, examples, strict=True):
            baseline_summary = _summary(example, low_values)
            patched_summary = _summary(example, patched_values)
            patched_loss = float(F.mse_loss(patched_summary, example["target_summary"]).item())
            label = int(example["target_label"].item())
            baseline_logits = baseline_summary @ classifier
            patched_logits = patched_summary @ classifier
            baseline_margin = baseline_logits[label] - torch.cat([baseline_logits[:label], baseline_logits[label + 1 :]]).max()
            patched_margin = patched_logits[label] - torch.cat([patched_logits[:label], patched_logits[label + 1 :]]).max()
            total += max(0.0, baseline_loss - patched_loss)
            total += 0.12 * max(0.0, float((patched_margin - baseline_margin).item()))
        scores[atom] = total / max(len(examples), 1)
    return scores


def _feature_persistence_scores(config: ToyMixedBitRouteAtomAllocatorConfig, problem: dict[str, torch.Tensor]) -> torch.Tensor:
    source_codes = problem["source_codes"]
    target_codes = problem["target_codes"]
    overlap = torch.minimum(source_codes, target_codes).sum(dim=-1) / torch.maximum(source_codes, target_codes).sum(
        dim=-1
    ).clamp_min(1e-8)
    cosine = (F.normalize(source_codes, dim=-1) * F.normalize(target_codes, dim=-1)).sum(dim=-1).clamp_min(0.0)
    raw = (0.65 * overlap + 0.35 * cosine).float()
    jitter = float(config.dictionary_noise) * torch.sin(torch.arange(config.atoms, dtype=torch.float32))
    return (raw + jitter).clamp(0.0, 1.0)


def _bits_from_scores(config: ToyMixedBitRouteAtomAllocatorConfig, scores: torch.Tensor) -> torch.Tensor:
    high_count = _target_high_count(config)
    bits = torch.full((config.atoms,), config.low_bits, dtype=torch.long)
    if high_count > 0:
        bits[torch.topk(scores, k=high_count, largest=True).indices] = config.high_bits
    return bits


def _selector_scores(
    config: ToyMixedBitRouteAtomAllocatorConfig,
    *,
    problem: dict[str, torch.Tensor],
    calibration: Sequence[dict[str, torch.Tensor]],
    seed_offset: int,
) -> dict[str, torch.Tensor]:
    mean_activation = _stack("activations", calibration).mean(dim=0)
    atom_values = problem["atom_values"]
    low_error = (atom_values - _symmetric_quantize(atom_values, config.low_bits)).pow(2).mean(dim=-1)
    exact = _exact_patch_scores(calibration, config=config, problem=problem)
    persistence = _feature_persistence_scores(config, problem)
    random = torch.rand(config.atoms, generator=_make_generator(config.seed + seed_offset))
    return {
        "uniform_3_bit": torch.zeros(config.atoms),
        "uniform_4_bit": torch.zeros(config.atoms),
        "quant_error_target_bpw_allocator": mean_activation * low_error,
        "exact_patch_target_bpw_allocator": exact,
        "universal_feature_persistence_allocator": mean_activation
        * (1.15 * persistence + 0.65 * low_error.sqrt() + 0.40 * persistence * low_error.sqrt()),
        "random_allocator": random,
        "oracle_allocator": exact,
        "_exact_patch_scores": exact,
        "_feature_persistence": persistence,
        "_quant_error": low_error,
    }


def _method_bits(config: ToyMixedBitRouteAtomAllocatorConfig, method: str, scores: dict[str, torch.Tensor]) -> torch.Tensor:
    if method == "uniform_3_bit":
        return torch.full((config.atoms,), config.low_bits, dtype=torch.long)
    if method == "uniform_4_bit":
        return torch.full((config.atoms,), config.mid_bits, dtype=torch.long)
    return _bits_from_scores(config, scores[method])


def _evaluate_method(
    method: str,
    *,
    config: ToyMixedBitRouteAtomAllocatorConfig,
    problem: dict[str, torch.Tensor],
    examples: Sequence[dict[str, torch.Tensor]],
    bit_allocation: torch.Tensor,
    selector_score: torch.Tensor,
    exact_scores: torch.Tensor,
    feature_persistence: torch.Tensor,
    quant_error: torch.Tensor,
    baseline_correct: Sequence[bool] | None,
) -> tuple[dict[str, Any], list[bool]]:
    quantized_values = _quantize_by_atom_bits(problem["atom_values"], bit_allocation)
    correct: list[bool] = []
    losses: list[float] = []
    for example in examples:
        recon = _summary(example, quantized_values)
        pred = int(torch.argmax(recon @ problem["classifier"]).item())
        correct.append(pred == int(example["target_label"].item()))
        losses.append(float(F.mse_loss(recon, example["target_summary"]).item()))

    if baseline_correct is None:
        help_rate = 0.0
        harm_rate = 0.0
    else:
        help_count = sum(bool(now) and not bool(base) for now, base in zip(correct, baseline_correct, strict=True))
        harm_count = sum((not bool(now)) and bool(base) for now, base in zip(correct, baseline_correct, strict=True))
        help_rate = float(help_count / max(len(correct), 1))
        harm_rate = float(harm_count / max(len(correct), 1))

    high_mask = bit_allocation.eq(config.high_bits)
    outlier_mask = problem["outlier_mask"]
    exact_high_mask = _topk_mask(exact_scores, int(high_mask.sum().item()))
    persistence_high_mask = _topk_mask(feature_persistence, int(high_mask.sum().item()))
    achieved_bpw = float(bit_allocation.float().mean().item())
    row = {
        "method": method,
        "accuracy": float(sum(correct) / max(len(correct), 1)),
        "mse": float(sum(losses) / max(len(losses), 1)),
        "achieved_bpw": achieved_bpw,
        "target_bpw": float(config.target_bpw) if method not in {"uniform_3_bit", "uniform_4_bit"} else achieved_bpw,
        "bit_allocation_histogram": _bit_histogram(bit_allocation),
        "high_bit_atoms": int(high_mask.sum().item()),
        "bytes_proxy": float(_bytes_proxy(bit_allocation, config.dim)),
        "compute_proxy": float(config.atoms * config.dim + high_mask.float().sum().item() * config.dim * 0.30),
        "patch_rank_correlation": _rank_correlation(selector_score, exact_scores),
        "exact_patch_overlap": _mask_overlap(high_mask, exact_high_mask),
        "feature_persistence_overlap": _mask_overlap(high_mask, persistence_high_mask),
        "mean_feature_persistence": float(feature_persistence[high_mask].mean().item()) if bool(high_mask.any()) else 0.0,
        "mean_quant_error": float(quant_error[high_mask].mean().item()) if bool(high_mask.any()) else 0.0,
        "outlier_protection": _mask_overlap(high_mask, outlier_mask),
        "help_vs_uniform_3_bit": help_rate,
        "harm_vs_uniform_3_bit": harm_rate,
    }
    return row, correct


def run_experiment(config: ToyMixedBitRouteAtomAllocatorConfig) -> dict[str, Any]:
    problem, calibration, test = _build_dataset(config)
    scores = _selector_scores(config, problem=problem, calibration=calibration, seed_offset=31_000)
    stability_scores = _selector_scores(config, problem=problem, calibration=calibration, seed_offset=41_000)
    test_oracle_scores = _exact_patch_scores(test, config=config, problem=problem)
    scores["oracle_allocator"] = test_oracle_scores
    stability_scores["oracle_allocator"] = test_oracle_scores

    rows: list[dict[str, Any]] = []
    baseline_correct: list[bool] | None = None
    baseline_accuracy = 0.0
    baseline_mse = 0.0
    for method in METHODS:
        bits = _method_bits(config, method, scores)
        row, correct = _evaluate_method(
            method,
            config=config,
            problem=problem,
            examples=test,
            bit_allocation=bits,
            selector_score=scores[method],
            exact_scores=scores["_exact_patch_scores"],
            feature_persistence=scores["_feature_persistence"],
            quant_error=scores["_quant_error"],
            baseline_correct=baseline_correct,
        )
        stable_bits = _method_bits(config, method, stability_scores)
        high_mask = bits.eq(config.high_bits)
        stable_high_mask = stable_bits.eq(config.high_bits)
        union = float((high_mask | stable_high_mask).float().sum().item())
        row["stability"] = float((high_mask & stable_high_mask).float().sum().item() / max(union, 1.0))
        if method == "uniform_3_bit":
            baseline_correct = correct
            baseline_accuracy = row["accuracy"]
            baseline_mse = row["mse"]
            row["accuracy_delta_vs_uniform_3_bit"] = 0.0
            row["mse_delta_vs_uniform_3_bit"] = 0.0
        else:
            row["accuracy_delta_vs_uniform_3_bit"] = float(row["accuracy"] - baseline_accuracy)
            row["mse_delta_vs_uniform_3_bit"] = float(row["mse"] - baseline_mse)
        rows.append(row)

    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
        "notes": [
            "All methods keep the same route atoms; the ablation isolates bit allocation under a target average bpw.",
            "Exact patch scores are calibration-only single-atom high-bit gains; oracle uses held-out exact patch gains.",
        ],
    }


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    lines = [
        "# Toy Mixed-Bit Route-Atom Allocator",
        "",
        "- EXL2/AWQ-style allocator toy: every route atom remains active, and only per-atom precision changes.",
        "- Mixed allocators target the configured average bpw by assigning high bits to selected route atoms and low bits elsewhere.",
        "- Patch-rank correlation measures agreement with calibration exact single-atom high-bit gains.",
        "",
        "| Method | Accuracy | Acc delta | MSE | MSE delta | Achieved bpw | Bit histogram | Patch-rank corr | Outlier protection | Exact overlap | Feature overlap | Stability | Bytes proxy | Compute proxy | Help vs 3-bit | Harm vs 3-bit |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {method} | {accuracy} | {accuracy_delta_vs_uniform_3_bit} | {mse} | {mse_delta_vs_uniform_3_bit} | {achieved_bpw} | `{hist}` | {patch_rank_correlation} | {outlier_protection} | {exact_patch_overlap} | {feature_persistence_overlap} | {stability} | {bytes_proxy} | {compute_proxy} | {help_vs_uniform_3_bit} | {harm_vs_uniform_3_bit} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                accuracy_delta_vs_uniform_3_bit=fmt(row["accuracy_delta_vs_uniform_3_bit"]),
                mse=fmt(row["mse"]),
                mse_delta_vs_uniform_3_bit=fmt(row["mse_delta_vs_uniform_3_bit"]),
                achieved_bpw=fmt(row["achieved_bpw"]),
                hist=json.dumps(row["bit_allocation_histogram"], sort_keys=True),
                patch_rank_correlation=fmt(row["patch_rank_correlation"]),
                outlier_protection=fmt(row["outlier_protection"]),
                exact_patch_overlap=fmt(row["exact_patch_overlap"]),
                feature_persistence_overlap=fmt(row["feature_persistence_overlap"]),
                stability=fmt(row["stability"]),
                bytes_proxy=fmt(row["bytes_proxy"]),
                compute_proxy=fmt(row["compute_proxy"]),
                help_vs_uniform_3_bit=fmt(row["help_vs_uniform_3_bit"]),
                harm_vs_uniform_3_bit=fmt(row["harm_vs_uniform_3_bit"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/query_pool_toy_20260421/mixed_bit_route_atom_allocator_20260421.json"))
    parser.add_argument("--output-md", type=pathlib.Path, default=pathlib.Path("results/query_pool_toy_20260421/mixed_bit_route_atom_allocator_20260421.md"))
    parser.add_argument("--seed", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.seed)
    parser.add_argument("--calibration-examples", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.calibration_examples)
    parser.add_argument("--test-examples", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.test_examples)
    parser.add_argument("--atoms", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.atoms)
    parser.add_argument("--dim", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.dim)
    parser.add_argument("--classes", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.classes)
    parser.add_argument("--universal-features", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.universal_features)
    parser.add_argument("--signal-atoms", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.signal_atoms)
    parser.add_argument("--outlier-atoms", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.outlier_atoms)
    parser.add_argument("--low-bits", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.low_bits)
    parser.add_argument("--mid-bits", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.mid_bits)
    parser.add_argument("--high-bits", type=int, default=ToyMixedBitRouteAtomAllocatorConfig.high_bits)
    parser.add_argument("--target-bpw", type=float, default=ToyMixedBitRouteAtomAllocatorConfig.target_bpw)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    config = ToyMixedBitRouteAtomAllocatorConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        atoms=args.atoms,
        dim=args.dim,
        classes=args.classes,
        universal_features=args.universal_features,
        signal_atoms=args.signal_atoms,
        outlier_atoms=args.outlier_atoms,
        low_bits=args.low_bits,
        mid_bits=args.mid_bits,
        high_bits=args.high_bits,
        target_bpw=args.target_bpw,
    )
    payload = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown_summary(payload, args.output_md)
    return payload


if __name__ == "__main__":
    main()
