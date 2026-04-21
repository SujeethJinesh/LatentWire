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
    "raw_activation_protect",
    "quant_error_protect",
    "exact_patch_effect_protect",
    "universal_dictionary_persistence_protect",
    "random_protect",
    "utility_oracle_protect",
)


@dataclass(frozen=True)
class ToyUniversalDictionaryFrontierConfig:
    seed: int = 0
    calibration_examples: int = 160
    test_examples: int = 192
    atoms: int = 30
    dim: int = 20
    classes: int = 4
    universal_features: int = 18
    signal_atoms: int = 8
    distractor_atoms: int = 10
    protected_atoms: int = 6
    keep_fraction: float = 0.70
    low_bits: int = 2
    high_bits: int = 8
    signal_scale: float = 2.7
    distractor_scale: float = 7.0
    bridge_noise: float = 0.10
    activation_noise: float = 0.030
    calibration_noise: float = 0.020
    dictionary_noise: float = 0.065
    cost_jitter: float = 0.12


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


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


def _make_mask(indices: torch.Tensor, atoms: int) -> torch.Tensor:
    mask = torch.zeros(atoms, dtype=torch.bool, device=indices.device)
    if indices.numel() > 0:
        mask[indices.long()] = True
    return mask


def _topk_mask(scores: torch.Tensor, k: int, candidate_mask: torch.Tensor | None = None) -> torch.Tensor:
    masked = scores.clone()
    if candidate_mask is not None:
        masked[~candidate_mask] = float("-inf")
    available = int(torch.isfinite(masked).sum().item())
    k = max(0, min(int(k), available))
    if k == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    indices = torch.topk(masked, k=k, largest=True).indices.sort().values
    return _make_mask(indices, scores.numel())


def _select_keep_mask(scores: torch.Tensor, atom_cost: torch.Tensor, *, keep_fraction: float) -> torch.Tensor:
    budget = float(keep_fraction) * float(atom_cost.sum().item())
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


def _rank_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return 0.0
    ar = torch.argsort(torch.argsort(a.float())).float()
    br = torch.argsort(torch.argsort(b.float())).float()
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = ar.norm() * br.norm()
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float((ar @ br / denom).clamp(-1.0, 1.0).item())


def _mask_overlap(selected: torch.Tensor, oracle: torch.Tensor) -> float:
    selected_set = set(int(i) for i in torch.where(selected)[0].tolist())
    oracle_set = set(int(i) for i in torch.where(oracle)[0].tolist())
    if not oracle_set:
        return 1.0 if not selected_set else 0.0
    return len(selected_set & oracle_set) / float(len(oracle_set))


def _make_problem(config: ToyUniversalDictionaryFrontierConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    if config.classes > config.dim:
        raise ValueError("classes must be <= dim")
    if config.signal_atoms + config.distractor_atoms > config.atoms:
        raise ValueError("signal_atoms + distractor_atoms must be <= atoms")
    if config.classes * 2 > config.universal_features:
        raise ValueError("universal_features must be at least 2 * classes")

    prototypes = _normalize_rows(torch.randn(config.classes, config.dim, generator=gen, dtype=torch.float32))
    classifier = prototypes.T.contiguous()

    source_dictionary = _normalize_rows(
        torch.randn(config.universal_features, config.dim, generator=gen, dtype=torch.float32)
    )
    target_dictionary = source_dictionary + float(config.bridge_noise) * torch.randn(
        config.universal_features, config.dim, generator=gen, dtype=torch.float32
    )
    target_dictionary = _normalize_rows(target_dictionary)

    source_codes = 0.010 * torch.rand(config.atoms, config.universal_features, generator=gen)
    target_codes = 0.010 * torch.rand(config.atoms, config.universal_features, generator=gen)
    utility = torch.zeros(config.atoms, dtype=torch.float32)
    atom_class = torch.full((config.atoms,), -1, dtype=torch.long)

    for atom in range(config.signal_atoms):
        cls = atom % config.classes
        feature_a = cls
        feature_b = config.classes + (atom // config.classes)
        strength = 1.45 + 0.06 * (config.signal_atoms - atom)
        source_codes[atom, feature_a] = strength
        target_codes[atom, feature_a] = strength + 0.04
        source_codes[atom, feature_b] = 0.55
        target_codes[atom, feature_b] = 0.58
        utility[atom] = 1.55 + 0.11 * (config.signal_atoms - atom)
        atom_class[atom] = cls

    start = config.signal_atoms
    end = start + config.distractor_atoms
    for pos, atom in enumerate(range(start, end)):
        source_feature = config.classes * 2 + pos % max(1, config.universal_features - config.classes * 2)
        target_feature = config.classes * 2 + (pos * 5 + 3) % max(1, config.universal_features - config.classes * 2)
        source_codes[atom, source_feature] = 2.0 + 0.08 * pos
        target_codes[atom, target_feature] = 2.0 + 0.08 * pos
        utility[atom] = -0.62 - 0.03 * pos

    if end < config.atoms:
        source_codes[end:] += 0.12 * torch.rand(config.atoms - end, config.universal_features, generator=gen)
        target_codes[end:] += 0.12 * torch.rand(config.atoms - end, config.universal_features, generator=gen)
        utility[end:] = 0.05 * torch.randn(config.atoms - end, generator=gen)

    atom_values = target_codes @ target_dictionary
    for atom in range(config.signal_atoms):
        cls = int(atom_class[atom].item())
        atom_values[atom] = atom_values[atom] + float(config.signal_scale) * prototypes[cls]
    for atom in range(start, end):
        raw = atom_values[atom] + float(config.distractor_scale) * torch.randn(config.dim, generator=gen)
        projection = prototypes.T @ (prototypes @ raw)
        orthogonal = raw - projection
        atom_values[atom] = float(config.distractor_scale) * orthogonal / orthogonal.norm().clamp_min(1e-8)

    atom_cost = 1.0 + 0.18 * torch.arange(config.atoms, dtype=torch.float32) / max(float(config.atoms - 1), 1.0)
    atom_cost = atom_cost + float(config.cost_jitter) * torch.rand(config.atoms, generator=gen)

    return {
        "atom_values": atom_values.float(),
        "classifier": classifier.float(),
        "utility": utility.float(),
        "atom_cost": atom_cost.float(),
        "atom_class": atom_class,
        "source_codes": source_codes.float(),
        "target_codes": target_codes.float(),
        "source_dictionary": source_dictionary.float(),
        "target_dictionary": target_dictionary.float(),
    }


def _make_examples(
    config: ToyUniversalDictionaryFrontierConfig,
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
        activations = 0.020 + float(config.activation_noise) * torch.rand(config.atoms, generator=gen)
        signal_match = atom_class.eq(label)
        signal_other = atom_class.ge(0) & ~signal_match
        activations[signal_match] += 1.68 + 0.30 * torch.rand(int(signal_match.sum().item()), generator=gen)
        activations[signal_other] += 0.11 * torch.rand(int(signal_other.sum().item()), generator=gen)

        distractor_start = config.signal_atoms
        distractor_end = config.signal_atoms + config.distractor_atoms
        activations[distractor_start:distractor_end] += 1.30 + 0.50 * torch.rand(
            config.distractor_atoms, generator=gen
        )
        activations = activations + float(config.calibration_noise) * torch.randn(config.atoms, generator=gen)
        activations = activations.clamp_min(0.0)
        activations = activations / activations.sum().clamp_min(1e-8)

        target_summary = torch.einsum("a,ad->d", activations, problem["atom_values"])
        target_summary = target_summary / activations.sum().clamp_min(1e-8)
        target_label = torch.argmax(target_summary @ problem["classifier"], dim=-1)
        examples.append(
            {
                "activations": activations.float(),
                "target_summary": target_summary.float(),
                "target_label": target_label.long(),
                "verifier_score": (activations * utility).float(),
            }
        )
    return examples


def _build_dataset(
    config: ToyUniversalDictionaryFrontierConfig,
) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]]]:
    problem = _make_problem(config)
    calibration = _make_examples(config, count=config.calibration_examples, seed_offset=13_000, problem=problem)
    test = _make_examples(config, count=config.test_examples, seed_offset=23_000, problem=problem)
    return problem, calibration, test


def _stack(field: str, examples: Sequence[dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.stack([example[field] for example in examples], dim=0)


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


def _exact_patch_scores(
    examples: Sequence[dict[str, torch.Tensor]],
    *,
    atom_values: torch.Tensor,
    classifier: torch.Tensor,
    keep_mask: torch.Tensor,
    low_bits: int,
    high_bits: int,
) -> torch.Tensor:
    scores = torch.zeros(atom_values.shape[0], dtype=torch.float32)
    low_protected = torch.zeros_like(keep_mask)
    low_recon = [
        _predict_summary(
            example,
            atom_values=atom_values,
            keep_mask=keep_mask,
            protected_mask=low_protected,
            low_bits=low_bits,
            high_bits=high_bits,
        )
        for example in examples
    ]
    for atom in torch.where(keep_mask)[0].tolist():
        protected = torch.zeros_like(keep_mask)
        protected[atom] = True
        total_gain = 0.0
        for baseline_summary, example in zip(low_recon, examples, strict=True):
            patched = _predict_summary(
                example,
                atom_values=atom_values,
                keep_mask=keep_mask,
                protected_mask=protected,
                low_bits=low_bits,
                high_bits=high_bits,
            )
            baseline_loss = F.mse_loss(baseline_summary, example["target_summary"]).item()
            patched_loss = F.mse_loss(patched, example["target_summary"]).item()
            label = int(example["target_label"].item())
            baseline_logits = baseline_summary @ classifier
            patched_logits = patched @ classifier
            baseline_margin = baseline_logits[label] - torch.cat(
                [baseline_logits[:label], baseline_logits[label + 1 :]]
            ).max()
            patched_margin = patched_logits[label] - torch.cat([patched_logits[:label], patched_logits[label + 1 :]]).max()
            # Protect atoms that repair the downstream class margin, using MSE gain only as a tie-breaker.
            total_gain += max(0.0, float((patched_margin - baseline_margin).item()))
            total_gain += 0.20 * max(0.0, baseline_loss - patched_loss)
        scores[atom] = float(total_gain / max(len(examples), 1))
    return scores


def _selector_masks(
    config: ToyUniversalDictionaryFrontierConfig,
    *,
    problem: dict[str, torch.Tensor],
    calibration: Sequence[dict[str, torch.Tensor]],
    seed_offset: int,
) -> dict[str, dict[str, torch.Tensor]]:
    activations = _stack("activations", calibration)
    mean_activation = activations.mean(dim=0)
    atom_values = problem["atom_values"]
    atom_cost = problem["atom_cost"]
    utility = problem["utility"]

    quant_error = ((atom_values - _symmetric_quantize(atom_values, config.low_bits)) ** 2).mean(dim=-1)
    keep_scores = mean_activation * (1.0 + 0.35 * quant_error.sqrt())
    keep_mask = _select_keep_mask(keep_scores, atom_cost, keep_fraction=config.keep_fraction)

    exact_scores = _exact_patch_scores(
        calibration,
        atom_values=atom_values,
        classifier=problem["classifier"],
        keep_mask=keep_mask,
        low_bits=config.low_bits,
        high_bits=config.high_bits,
    )
    source_codes = problem["source_codes"]
    target_codes = problem["target_codes"]
    code_overlap = torch.minimum(source_codes, target_codes).sum(dim=-1) / torch.maximum(source_codes, target_codes).sum(
        dim=-1
    ).clamp_min(1e-8)
    source_norm = F.normalize(source_codes, dim=-1)
    target_norm = F.normalize(target_codes, dim=-1)
    code_cosine = (source_norm * target_norm).sum(dim=-1).clamp_min(0.0)
    feature_persistence = (0.65 * code_overlap + 0.35 * code_cosine).float()
    learned_persistence = (feature_persistence + float(config.dictionary_noise) * torch.sin(torch.arange(config.atoms))).clamp(
        0.0, 1.0
    )

    random_gen = _make_generator(config.seed + seed_offset)
    random_scores = torch.rand(config.atoms, generator=random_gen)
    utility_positive = (utility.clamp_min(0.0) * mean_activation).float()

    protected_scores = {
        "prune_uniform_quant": torch.full((config.atoms,), float("-inf")),
        "raw_activation_protect": mean_activation,
        "quant_error_protect": mean_activation * quant_error,
        "exact_patch_effect_protect": exact_scores,
        "universal_dictionary_persistence_protect": mean_activation
        * (1.50 * learned_persistence + 0.55 * quant_error + 0.25 * learned_persistence * quant_error.sqrt()),
        "random_protect": random_scores,
        "utility_oracle_protect": utility_positive,
    }
    masks: dict[str, dict[str, torch.Tensor]] = {}
    for method, scores in protected_scores.items():
        if method == "prune_uniform_quant":
            protected = torch.zeros(config.atoms, dtype=torch.bool)
        else:
            protected = _topk_mask(scores, config.protected_atoms, keep_mask)
        masks[method] = {
            "keep_mask": keep_mask.clone(),
            "protected_mask": protected,
            "selector_scores": scores.float(),
            "patch_scores": exact_scores.float(),
            "feature_persistence": learned_persistence.float(),
            "mean_activation": mean_activation.float(),
            "quant_error": quant_error.float(),
        }
    return masks


def _evaluate_method(
    method: str,
    examples: Sequence[dict[str, torch.Tensor]],
    *,
    config: ToyUniversalDictionaryFrontierConfig,
    problem: dict[str, torch.Tensor],
    keep_mask: torch.Tensor,
    protected_mask: torch.Tensor,
    selector_scores: torch.Tensor,
    patch_scores: torch.Tensor,
    feature_persistence: torch.Tensor,
    protected_oracle_mask: torch.Tensor,
    baseline_correct: Sequence[bool] | None,
) -> tuple[dict[str, Any], list[bool]]:
    correct_flags: list[bool] = []
    losses: list[float] = []
    classifier = problem["classifier"]

    for example in examples:
        summary = _predict_summary(
            example,
            atom_values=problem["atom_values"],
            keep_mask=keep_mask,
            protected_mask=protected_mask,
            low_bits=config.low_bits,
            high_bits=config.high_bits,
        )
        pred = int(torch.argmax(summary @ classifier).item())
        target = int(example["target_label"].item())
        correct_flags.append(pred == target)
        losses.append(float(F.mse_loss(summary, example["target_summary"]).item()))

    if baseline_correct is None:
        help_rate = 0.0
        harm_rate = 0.0
    else:
        help_rate = sum(bool(now) and not bool(base) for now, base in zip(correct_flags, baseline_correct, strict=True))
        harm_rate = sum((not bool(now)) and bool(base) for now, base in zip(correct_flags, baseline_correct, strict=True))
        help_rate = float(help_rate / max(len(correct_flags), 1))
        harm_rate = float(harm_rate / max(len(correct_flags), 1))

    protected_count = int(protected_mask.sum().item())
    kept_count = int(keep_mask.sum().item())
    utility = problem["utility"]
    helpful_mask = utility > 0
    pruned_mask = ~keep_mask
    selected_persistence = float(feature_persistence[protected_mask].mean().item()) if protected_count else 0.0
    all_persistence = float(feature_persistence.mean().item())
    top_persistent = _topk_mask(feature_persistence, max(protected_count, 1), keep_mask)

    row = {
        "method": method,
        "accuracy": float(sum(correct_flags) / max(len(correct_flags), 1)),
        "mse": float(sum(losses) / max(len(losses), 1)),
        "prune_rate": float(pruned_mask.float().mean().item()),
        "kept_rate": float(keep_mask.float().mean().item()),
        "protected_rate": float(protected_mask.float().mean().item()),
        "kept_atoms": kept_count,
        "protected_atoms": protected_count,
        "bytes_proxy": float(
            _estimate_uniform_bytes(config.atoms, config.dim, config.low_bits)
            if protected_count == 0
            else _estimate_mixed_bytes(config.atoms, config.dim, config.low_bits, config.high_bits, protected_count)
        ),
        "compute_proxy": float(kept_count * config.dim + protected_count * config.dim * 0.22),
        "feature_overlap_persistence": selected_persistence,
        "feature_overlap_lift": float(selected_persistence - all_persistence),
        "top_persistent_feature_preservation": _mask_overlap(protected_mask, top_persistent),
        "protected_oracle_preservation_rate": _mask_overlap(protected_mask, protected_oracle_mask),
        "patch_rank_correlation": _rank_correlation(selector_scores[keep_mask], patch_scores[keep_mask])
        if bool(keep_mask.any())
        else 0.0,
        "protection_precision_rate": float(
            (protected_mask & protected_oracle_mask).float().sum().item() / max(float(protected_count), 1.0)
        ),
        "missed_help_rate": float(
            (pruned_mask & helpful_mask).float().sum().item() / max(float(helpful_mask.float().sum().item()), 1.0)
        ),
        "false_prune_rate": float(
            (pruned_mask & helpful_mask).float().sum().item() / max(float(pruned_mask.float().sum().item()), 1.0)
        ),
        "help_vs_prune_uniform_quant": help_rate,
        "harm_vs_prune_uniform_quant": harm_rate,
    }
    return row, correct_flags


def run_experiment(config: ToyUniversalDictionaryFrontierConfig) -> dict[str, Any]:
    problem, calibration, test = _build_dataset(config)
    masks = _selector_masks(config, problem=problem, calibration=calibration, seed_offset=31_000)
    stability_masks = _selector_masks(config, problem=problem, calibration=calibration, seed_offset=41_000)

    exact_scores = masks["exact_patch_effect_protect"]["patch_scores"]
    keep_mask = masks["exact_patch_effect_protect"]["keep_mask"]
    protected_oracle_mask = _topk_mask(exact_scores, config.protected_atoms, keep_mask)

    rows: list[dict[str, Any]] = []
    baseline_correct: list[bool] | None = None
    baseline_mse = 0.0
    baseline_accuracy = 0.0

    for method in METHODS:
        method_masks = masks[method]
        row, correct = _evaluate_method(
            method,
            test,
            config=config,
            problem=problem,
            keep_mask=method_masks["keep_mask"],
            protected_mask=method_masks["protected_mask"],
            selector_scores=method_masks["selector_scores"],
            patch_scores=method_masks["patch_scores"],
            feature_persistence=method_masks["feature_persistence"],
            protected_oracle_mask=protected_oracle_mask,
            baseline_correct=baseline_correct,
        )
        if method == "prune_uniform_quant":
            baseline_correct = correct
            baseline_mse = row["mse"]
            baseline_accuracy = row["accuracy"]
            row["accuracy_delta_vs_prune_uniform_quant"] = 0.0
            row["mse_delta_vs_prune_uniform_quant"] = 0.0
        else:
            row["accuracy_delta_vs_prune_uniform_quant"] = float(row["accuracy"] - baseline_accuracy)
            row["mse_delta_vs_prune_uniform_quant"] = float(row["mse"] - baseline_mse)
        primary = method_masks["protected_mask"]
        secondary = stability_masks[method]["protected_mask"]
        union = float((primary | secondary).float().sum().item())
        row["selector_stability"] = float((primary & secondary).float().sum().item() / max(union, 1.0))
        rows.append(row)

    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
        "notes": [
            "Universal dictionary persistence scores approximate shared SAE features that survive source/target views.",
            "Exact patch-effect is the calibration oracle; utility oracle isolates semantic helpfulness from compression criticality.",
        ],
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Toy Universal Dictionary Frontier",
        "",
        "A deterministic protected-frontier toy for SAE/universal-dictionary-inspired selection.",
        "",
        "| Method | Accuracy | Acc delta | MSE | MSE delta | Feature persistence | Patch-rank corr | Selector stability | Protected-oracle preservation | Bytes proxy | Compute proxy | Help vs prune-uniform | Harm vs prune-uniform |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {accuracy_delta_vs_prune_uniform_quant} | {mse} | {mse_delta_vs_prune_uniform_quant} | {feature_overlap_persistence} | {patch_rank_correlation} | {selector_stability} | {protected_oracle_preservation_rate} | {bytes_proxy} | {compute_proxy} | {help_vs_prune_uniform_quant} | {harm_vs_prune_uniform_quant} |".format(
                method=row["method"],
                accuracy=_fmt(row["accuracy"]),
                accuracy_delta_vs_prune_uniform_quant=_fmt(row["accuracy_delta_vs_prune_uniform_quant"]),
                mse=_fmt(row["mse"]),
                mse_delta_vs_prune_uniform_quant=_fmt(row["mse_delta_vs_prune_uniform_quant"]),
                feature_overlap_persistence=_fmt(row["feature_overlap_persistence"]),
                patch_rank_correlation=_fmt(row["patch_rank_correlation"]),
                selector_stability=_fmt(row["selector_stability"]),
                protected_oracle_preservation_rate=_fmt(row["protected_oracle_preservation_rate"]),
                bytes_proxy=_fmt(row["bytes_proxy"]),
                compute_proxy=_fmt(row["compute_proxy"]),
                help_vs_prune_uniform_quant=_fmt(row["help_vs_prune_uniform_quant"]),
                harm_vs_prune_uniform_quant=_fmt(row["harm_vs_prune_uniform_quant"]),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: a useful shared dictionary selector should be more stable than random, preserve more exact patch-effect atoms than raw activation, and improve MSE without relying on a utility oracle.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--output-md", help="Markdown summary path. Defaults to the JSON path with .md suffix.")
    parser.add_argument("--seed", type=int, default=ToyUniversalDictionaryFrontierConfig.seed)
    parser.add_argument("--calibration-examples", type=int, default=ToyUniversalDictionaryFrontierConfig.calibration_examples)
    parser.add_argument("--test-examples", type=int, default=ToyUniversalDictionaryFrontierConfig.test_examples)
    parser.add_argument("--atoms", type=int, default=ToyUniversalDictionaryFrontierConfig.atoms)
    parser.add_argument("--dim", type=int, default=ToyUniversalDictionaryFrontierConfig.dim)
    parser.add_argument("--classes", type=int, default=ToyUniversalDictionaryFrontierConfig.classes)
    parser.add_argument("--universal-features", type=int, default=ToyUniversalDictionaryFrontierConfig.universal_features)
    parser.add_argument("--signal-atoms", type=int, default=ToyUniversalDictionaryFrontierConfig.signal_atoms)
    parser.add_argument("--distractor-atoms", type=int, default=ToyUniversalDictionaryFrontierConfig.distractor_atoms)
    parser.add_argument("--protected-atoms", type=int, default=ToyUniversalDictionaryFrontierConfig.protected_atoms)
    parser.add_argument("--keep-fraction", type=float, default=ToyUniversalDictionaryFrontierConfig.keep_fraction)
    parser.add_argument("--low-bits", type=int, default=ToyUniversalDictionaryFrontierConfig.low_bits)
    parser.add_argument("--high-bits", type=int, default=ToyUniversalDictionaryFrontierConfig.high_bits)
    args = parser.parse_args(argv)

    config = ToyUniversalDictionaryFrontierConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        atoms=args.atoms,
        dim=args.dim,
        classes=args.classes,
        universal_features=args.universal_features,
        signal_atoms=args.signal_atoms,
        distractor_atoms=args.distractor_atoms,
        protected_atoms=args.protected_atoms,
        keep_fraction=args.keep_fraction,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
    )
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown = pathlib.Path(args.output_md) if args.output_md else output.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(payload, markdown)
    return payload


if __name__ == "__main__":
    main()
