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
class ToySensitivityBudgetConfig:
    seed: int = 0
    calibration_examples: int = 96
    test_examples: int = 192
    slots: int = 12
    channels: int = 24
    classes: int = 4
    protected_slots: int = 3
    protected_channels: int = 4
    bits: int = 4
    signal_scale: float = 2.25
    noise_scale: float = 0.14
    outlier_scale: float = 7.5
    scenarios: tuple[str, ...] = ("aligned", "rotated", "outlier", "slot_permuted")


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _make_signal_indices(length: int, count: int, *, offset: int, step: int) -> torch.Tensor:
    count = max(1, min(int(count), int(length)))
    if count == length:
        return torch.arange(length, dtype=torch.long)
    indices = []
    seen = set()
    cursor = int(offset) % int(length)
    while len(indices) < count:
        candidate = cursor % length
        if candidate not in seen:
            indices.append(candidate)
            seen.add(candidate)
        cursor += step
    return torch.tensor(sorted(indices), dtype=torch.long)


def _entropy_from_probs(probs: torch.Tensor) -> float:
    probs = probs / probs.sum().clamp_min(1e-8)
    return float((-(probs * probs.clamp_min(1e-8).log()).sum()).item())


def _build_base_weights(config: ToySensitivityBudgetConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gen = _make_generator(config.seed)
    slot_a = _make_signal_indices(config.slots, config.protected_slots, offset=1, step=3)
    slot_b = _make_signal_indices(config.slots, config.protected_slots, offset=2, step=5)
    channel_a = _make_signal_indices(config.channels, config.protected_channels, offset=2, step=5)
    channel_b = _make_signal_indices(config.channels, config.protected_channels, offset=4, step=7)

    slot_profile_a = torch.zeros(config.slots, dtype=torch.float32)
    slot_profile_b = torch.zeros(config.slots, dtype=torch.float32)
    channel_profile_a = torch.zeros(config.channels, dtype=torch.float32)
    channel_profile_b = torch.zeros(config.channels, dtype=torch.float32)
    slot_profile_a[slot_a] = torch.linspace(1.35, 0.85, steps=slot_a.numel())
    slot_profile_b[slot_b] = torch.linspace(1.15, 0.75, steps=slot_b.numel())
    channel_profile_a[channel_a] = torch.linspace(1.40, 0.80, steps=channel_a.numel())
    channel_profile_b[channel_b] = torch.linspace(1.20, 0.70, steps=channel_b.numel())

    core_a = torch.outer(slot_profile_a, channel_profile_a)
    core_b = torch.outer(slot_profile_b, channel_profile_b)
    class_coefficients = torch.tensor(
        [
            [1.0, 0.3],
            [0.3, 1.0],
            [-1.0, 0.4],
            [0.4, -1.0],
        ],
        dtype=torch.float32,
    )
    if class_coefficients.shape[0] != config.classes:
        class_coefficients = torch.randn(config.classes, 2, generator=gen, dtype=torch.float32)
        class_coefficients = class_coefficients / class_coefficients.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

    weights = []
    for class_index in range(config.classes):
        weight = 0.02 * torch.randn(config.slots, config.channels, generator=gen, dtype=torch.float32)
        weight = weight + config.signal_scale * (
            class_coefficients[class_index, 0] * core_a + class_coefficients[class_index, 1] * core_b
        )
        weight = weight + 0.01 * torch.randn(config.slots, config.channels, generator=gen, dtype=torch.float32)
        weights.append(weight)
    return torch.stack(weights, dim=0), {
        "slot_a": slot_a,
        "slot_b": slot_b,
        "channel_a": channel_a,
        "channel_b": channel_b,
    }


def _transform_weights(
    weights: torch.Tensor,
    scenario: str,
    *,
    seed: int,
    config: ToySensitivityBudgetConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    transformed = weights.clone()
    meta: dict[str, torch.Tensor] = {}
    if scenario == "aligned":
        return transformed, meta
    if scenario == "rotated":
        rotation = _orthogonal_matrix(config.channels, _make_generator(seed + 101)).to(dtype=transformed.dtype)
        transformed = torch.einsum("ksc,cd->ksd", transformed, rotation)
        meta["rotation"] = rotation
        return transformed, meta
    if scenario == "outlier":
        scale = torch.ones(config.channels, dtype=transformed.dtype)
        outlier_channels = _make_signal_indices(config.channels, max(1, config.channels // 8), offset=3, step=4)
        outlier_slots = _make_signal_indices(config.slots, max(1, config.slots // 8), offset=2, step=3)
        scale[outlier_channels] = float(config.outlier_scale)
        transformed = transformed * scale.view(1, 1, -1)
        slot_scale = torch.ones(config.slots, dtype=transformed.dtype)
        slot_scale[outlier_slots] = float(config.outlier_scale) ** 0.5
        transformed = transformed * slot_scale.view(1, -1, 1)
        meta["outlier_channels"] = outlier_channels
        meta["outlier_slots"] = outlier_slots
        return transformed, meta
    if scenario == "slot_permuted":
        perm = torch.randperm(config.slots, generator=_make_generator(seed + 211))
        transformed = transformed[:, perm]
        meta["slot_perm"] = perm
        return transformed, meta
    raise ValueError(f"Unknown scenario: {scenario}")


def _make_split(
    *,
    weights: torch.Tensor,
    scenario: str,
    split: str,
    config: ToySensitivityBudgetConfig,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    base_seed = seed + (0 if split == "calibration" else 10_000) + sum(ord(ch) for ch in scenario)
    gen = _make_generator(base_seed)
    examples = config.calibration_examples if split == "calibration" else config.test_examples
    x = torch.randn(examples, config.slots, config.channels, generator=gen, dtype=torch.float32)
    x = x + config.noise_scale * torch.randn(x.shape, generator=gen, dtype=x.dtype)
    scenario_meta: dict[str, torch.Tensor] = {}
    if scenario == "outlier":
        outlier_channels = _make_signal_indices(config.channels, max(1, config.channels // 8), offset=3, step=4)
        outlier_slots = _make_signal_indices(config.slots, max(1, config.slots // 8), offset=2, step=3)
        x[:, :, outlier_channels] = x[:, :, outlier_channels] * float(config.outlier_scale)
        x[:, outlier_slots, :] = x[:, outlier_slots, :] * float(config.outlier_scale) ** 0.5
        scenario_meta["outlier_channels"] = outlier_channels
        scenario_meta["outlier_slots"] = outlier_slots
    elif scenario == "slot_permuted":
        perm = torch.randperm(config.slots, generator=_make_generator(seed + 211))
        x = x[:, perm]
        scenario_meta["slot_perm"] = perm
    elif scenario == "rotated":
        rotation = _orthogonal_matrix(config.channels, _make_generator(seed + 101)).to(dtype=x.dtype)
        x = torch.einsum("nsc,cd->nsd", x, rotation)
        scenario_meta["rotation"] = rotation
    labels = _classify(x, weights).argmax(dim=-1)
    return x, labels, scenario_meta


def _classify(x: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nsc,ksc->nk", x, prototypes)


def _accuracy(x: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor) -> tuple[float, torch.Tensor]:
    logits = _classify(x, prototypes)
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item()), pred


def _margin_drop_scores(
    x: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    axis: str,
) -> torch.Tensor:
    full_logits = _classify(x, prototypes)
    full_true = full_logits.gather(1, labels.view(-1, 1)).squeeze(1)
    full_other = full_logits.masked_fill(
        F.one_hot(labels, num_classes=full_logits.shape[-1]).bool(),
        float("-inf"),
    ).amax(dim=-1)
    full_margin = full_true - full_other
    scores = []
    if axis == "channel":
        for channel in range(x.shape[-1]):
            masked = x.clone()
            masked[:, :, channel] = 0.0
            logits = _classify(masked, prototypes)
            true = logits.gather(1, labels.view(-1, 1)).squeeze(1)
            other = logits.masked_fill(
                F.one_hot(labels, num_classes=logits.shape[-1]).bool(),
                float("-inf"),
            ).amax(dim=-1)
            scores.append(float((full_margin - (true - other)).mean().item()))
    elif axis == "slot":
        for slot in range(x.shape[1]):
            masked = x.clone()
            masked[:, slot, :] = 0.0
            logits = _classify(masked, prototypes)
            true = logits.gather(1, labels.view(-1, 1)).squeeze(1)
            other = logits.masked_fill(
                F.one_hot(labels, num_classes=logits.shape[-1]).bool(),
                float("-inf"),
            ).amax(dim=-1)
            scores.append(float((full_margin - (true - other)).mean().item()))
    else:
        raise ValueError(f"Unknown axis: {axis}")
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    return torch.clamp(score_tensor, min=0.0)


def _true_scores(prototypes: torch.Tensor, *, axis: str) -> torch.Tensor:
    if axis == "channel":
        return prototypes.abs().mean(dim=(0, 1))
    if axis == "slot":
        return prototypes.abs().mean(dim=(0, 2))
    raise ValueError(f"Unknown axis: {axis}")


def _uniform_scores(length: int) -> torch.Tensor:
    return torch.ones(length, dtype=torch.float32)


def _select_uniform(length: int, count: int) -> torch.Tensor:
    count = max(1, min(int(count), int(length)))
    if count >= length:
        return torch.arange(length, dtype=torch.long)
    indices = [(i * length) // count for i in range(count)]
    return torch.tensor(sorted(set(indices)), dtype=torch.long)


def _select_topk(scores: torch.Tensor, count: int) -> torch.Tensor:
    count = max(1, min(int(count), int(scores.numel())))
    return torch.topk(scores, k=count, largest=True).indices.sort().values


def _compress_tensor(
    x: torch.Tensor,
    *,
    protected_slots: torch.Tensor,
    protected_channels: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, int, float]:
    protected_mask = torch.zeros_like(x, dtype=torch.bool)
    if protected_slots.numel() > 0:
        protected_mask[:, protected_slots, :] = True
    if protected_channels.numel() > 0:
        protected_mask[:, :, protected_channels] = True
    protected_fraction = float(protected_mask.float().mean().item())

    quantized = x.clone()
    unprotected = ~protected_mask
    if unprotected.any():
        values = x[unprotected]
        qmax = float(2 ** (bits - 1) - 1)
        scale = values.abs().amax().clamp_min(1e-8) / qmax
        codes = torch.round(values / scale).clamp(-qmax, qmax)
        quantized[unprotected] = codes * scale

    protected_count = int(protected_mask.sum().item())
    total_count = int(x.numel())
    index_bytes = 1 if max(x.shape[0], x.shape[1], x.shape[2]) < 256 else 2
    bytes_estimate = (
        protected_count * 2
        + (total_count - protected_count) * bits / 8.0
        + int(protected_slots.numel() + protected_channels.numel()) * index_bytes
        + 4
    )
    return quantized, bytes_estimate, protected_fraction


def _entropy_from_scores(scores: torch.Tensor, *, uniform_fallback: bool = False) -> float:
    if uniform_fallback or float(scores.sum().item()) <= 0.0:
        probs = torch.ones_like(scores, dtype=torch.float32) / float(scores.numel())
    else:
        probs = scores.clamp_min(0.0)
        probs = probs / probs.sum().clamp_min(1e-8)
        if torch.isnan(probs).any() or float(probs.sum().item()) <= 0.0:
            probs = torch.ones_like(scores, dtype=torch.float32) / float(scores.numel())
    return _entropy_from_probs(probs)


def _outlier_mass(x: torch.Tensor) -> float:
    channel_energy = x.pow(2).mean(dim=(0, 1))
    slot_energy = x.pow(2).mean(dim=(0, 2))
    top_channels = max(1, x.shape[-1] // 8)
    top_slots = max(1, x.shape[1] // 8)
    channel_mass = channel_energy.topk(k=min(top_channels, channel_energy.numel())).values.sum() / channel_energy.sum().clamp_min(1e-8)
    slot_mass = slot_energy.topk(k=min(top_slots, slot_energy.numel())).values.sum() / slot_energy.sum().clamp_min(1e-8)
    return float(0.5 * (channel_mass + slot_mass).item())


def _selected_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_set = set(int(v) for v in lhs.tolist())
    rhs_set = set(int(v) for v in rhs.tolist())
    if not lhs_set and not rhs_set:
        return 1.0
    return float(len(lhs_set & rhs_set) / max(len(lhs_set | rhs_set), 1))


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.clamp_min(0.0)
    if float(scores.sum().item()) <= 0.0:
        return torch.ones_like(scores) / float(scores.numel())
    return scores / scores.sum().clamp_min(1e-8)


def _estimate_bytes(config: ToySensitivityBudgetConfig) -> int:
    protected_cells = config.protected_slots * config.channels + config.protected_channels * config.slots - (
        config.protected_slots * config.protected_channels
    )
    total_cells = config.slots * config.channels
    quantized_cells = total_cells - protected_cells
    index_bytes = 1 if max(config.slots, config.channels) < 256 else 2
    return int(protected_cells * 2 + quantized_cells * config.bits / 8.0 + (config.protected_slots + config.protected_channels) * index_bytes + 4)


def _evaluate_method(
    *,
    scenario: str,
    method: str,
    selected_slots: torch.Tensor,
    selected_channels: torch.Tensor,
    calibration_x: torch.Tensor,
    calibration_labels: torch.Tensor,
    test_x: torch.Tensor,
    test_labels: torch.Tensor,
    weights: torch.Tensor,
    oracle_slots: torch.Tensor,
    oracle_channels: torch.Tensor,
    config: ToySensitivityBudgetConfig,
) -> dict[str, Any]:
    compressed, bytes_estimate, protected_fraction = _compress_tensor(
        test_x,
        protected_slots=selected_slots,
        protected_channels=selected_channels,
        bits=config.bits,
    )
    accuracy, predictions = _accuracy(compressed, test_labels, weights)
    uniform_slots = _select_uniform(config.slots, config.protected_slots)
    uniform_channels = _select_uniform(config.channels, config.protected_channels)
    uniform_compressed, _, _ = _compress_tensor(
        test_x,
        protected_slots=uniform_slots,
        protected_channels=uniform_channels,
        bits=config.bits,
    )
    uniform_accuracy, uniform_predictions = _accuracy(uniform_compressed, test_labels, weights)
    help_rate = float(((predictions == test_labels) & (uniform_predictions != test_labels)).float().mean().item())
    harm_rate = float(((predictions != test_labels) & (uniform_predictions == test_labels)).float().mean().item())

    if method == "uniform_allocation":
        channel_scores = _uniform_scores(config.channels)
        slot_scores = _uniform_scores(config.slots)
    elif method == "sensitivity_protected":
        channel_scores = _margin_drop_scores(calibration_x, calibration_labels, weights, axis="channel")
        slot_scores = _margin_drop_scores(calibration_x, calibration_labels, weights, axis="slot")
    elif method == "oracle_allocation":
        channel_scores = _true_scores(weights, axis="channel")
        slot_scores = _true_scores(weights, axis="slot")
    else:
        raise ValueError(f"Unknown method: {method}")

    channel_entropy = _entropy_from_scores(channel_scores, uniform_fallback=method == "uniform_allocation")
    slot_entropy = _entropy_from_scores(slot_scores, uniform_fallback=method == "uniform_allocation")
    allocation_entropy = 0.5 * (channel_entropy + slot_entropy)
    outlier_mass = _outlier_mass(test_x)

    channel_overlap = _selected_overlap(selected_channels, oracle_channels)
    slot_overlap = _selected_overlap(selected_slots, oracle_slots)
    selected_mass = float(
        0.5
        * (
            channel_scores[selected_channels].sum().item() / channel_scores.sum().clamp_min(1e-8).item()
            + slot_scores[selected_slots].sum().item() / slot_scores.sum().clamp_min(1e-8).item()
        )
    )

    return {
        "scenario": scenario,
        "method": method,
        "accuracy": accuracy,
        "uniform_accuracy": uniform_accuracy,
        "accuracy_delta_vs_uniform": accuracy - uniform_accuracy,
        "help_rate_vs_uniform": help_rate,
        "harm_rate_vs_uniform": harm_rate,
        "protected_fraction": protected_fraction,
        "allocation_entropy": allocation_entropy,
        "outlier_mass": outlier_mass,
        "bytes_estimate": float(bytes_estimate),
        "protected_slots": int(selected_slots.numel()),
        "protected_channels": int(selected_channels.numel()),
        "selected_slot_indices": [int(v) for v in selected_slots.tolist()],
        "selected_channel_indices": [int(v) for v in selected_channels.tolist()],
        "channel_overlap_with_oracle": channel_overlap,
        "slot_overlap_with_oracle": slot_overlap,
        "selected_score_mass": selected_mass,
        "calibration_examples": int(calibration_x.shape[0]),
        "test_examples": int(test_x.shape[0]),
        "bits": int(config.bits),
    }


def run_experiment(config: ToySensitivityBudgetConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_weights, _ = _build_base_weights(config)
    for scenario in config.scenarios:
        weights, transform_meta = _transform_weights(base_weights, scenario, seed=config.seed, config=config)
        calibration_x, calibration_labels, _ = _make_split(
            weights=weights,
            scenario=scenario,
            split="calibration",
            config=config,
            seed=config.seed,
        )
        test_x, test_labels, _ = _make_split(
            weights=weights,
            scenario=scenario,
            split="test",
            config=config,
            seed=config.seed,
        )

        uniform_slots = _select_uniform(config.slots, config.protected_slots)
        uniform_channels = _select_uniform(config.channels, config.protected_channels)
        sensitivity_channel_scores = _margin_drop_scores(calibration_x, calibration_labels, weights, axis="channel")
        sensitivity_slot_scores = _margin_drop_scores(calibration_x, calibration_labels, weights, axis="slot")
        sensitivity_slots = _select_topk(sensitivity_slot_scores, config.protected_slots)
        sensitivity_channels = _select_topk(sensitivity_channel_scores, config.protected_channels)
        oracle_slots = _select_topk(_true_scores(weights, axis="slot"), config.protected_slots)
        oracle_channels = _select_topk(_true_scores(weights, axis="channel"), config.protected_channels)

        method_specs = (
            ("uniform_allocation", uniform_slots, uniform_channels),
            ("sensitivity_protected", sensitivity_slots, sensitivity_channels),
            ("oracle_allocation", oracle_slots, oracle_channels),
        )
        for method, slots, channels in method_specs:
            row = _evaluate_method(
                scenario=scenario,
                method=method,
                selected_slots=slots,
                selected_channels=channels,
                calibration_x=calibration_x,
                calibration_labels=calibration_labels,
                test_x=test_x,
                test_labels=test_labels,
                weights=weights,
                oracle_slots=oracle_slots,
                oracle_channels=oracle_channels,
                config=config,
            )
            if "rotation" in transform_meta:
                row["rotation_applied"] = True
            if "slot_perm" in transform_meta:
                row["slot_perm_applied"] = True
            if "outlier_slots" in transform_meta:
                row["outlier_slots_applied"] = [int(v) for v in transform_meta["outlier_slots"].tolist()]
            if "outlier_channels" in transform_meta:
                row["outlier_channels_applied"] = [int(v) for v in transform_meta["outlier_channels"].tolist()]
            rows.append(row)
    return rows


def _group_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for scenario in sorted({row["scenario"] for row in rows}):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        uniform = next(row for row in scenario_rows if row["method"] == "uniform_allocation")
        best = max(scenario_rows, key=lambda row: row["accuracy"])
        grouped.append(
            {
                "scenario": scenario,
                "best_method": best["method"],
                "best_accuracy": best["accuracy"],
                "uniform_accuracy": uniform["accuracy"],
                "uniform_bytes_estimate": uniform["bytes_estimate"],
                "mean_accuracy": float(sum(row["accuracy"] for row in scenario_rows) / len(scenario_rows)),
                "mean_help_rate_vs_uniform": float(sum(row["help_rate_vs_uniform"] for row in scenario_rows) / len(scenario_rows)),
                "mean_harm_rate_vs_uniform": float(sum(row["harm_rate_vs_uniform"] for row in scenario_rows) / len(scenario_rows)),
            }
        )
    return grouped


def write_jsonl(rows: Sequence[dict[str, Any]], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_markdown_summary(rows: Sequence[dict[str, Any]], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return "[" + ", ".join(str(item) for item in value) + "]"
        return f"{float(value):.4f}"

    lines = [
        "# Toy Sensitivity Budget Bridge",
        "",
        "Matched-budget comparison across allocation policies.",
        "",
        "| Scenario | Method | Accuracy | Uniform acc. | Δ acc. | Help vs uniform | Harm vs uniform | Protected fraction | Allocation entropy | Outlier mass | Bytes estimate | Selected slots | Selected channels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {method} | {accuracy} | {uniform_accuracy} | {accuracy_delta_vs_uniform} | {help_rate_vs_uniform} | {harm_rate_vs_uniform} | {protected_fraction} | {allocation_entropy} | {outlier_mass} | {bytes_estimate} | {selected_slot_indices} | {selected_channel_indices} |".format(
                scenario=row["scenario"],
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                uniform_accuracy=fmt(row["uniform_accuracy"]),
                accuracy_delta_vs_uniform=fmt(row["accuracy_delta_vs_uniform"]),
                help_rate_vs_uniform=fmt(row["help_rate_vs_uniform"]),
                harm_rate_vs_uniform=fmt(row["harm_rate_vs_uniform"]),
                protected_fraction=fmt(row["protected_fraction"]),
                allocation_entropy=fmt(row["allocation_entropy"]),
                outlier_mass=fmt(row["outlier_mass"]),
                bytes_estimate=fmt(row["bytes_estimate"]),
                selected_slot_indices=fmt(row["selected_slot_indices"]),
                selected_channel_indices=fmt(row["selected_channel_indices"]),
            )
        )
    grouped = _group_rows(rows)
    lines.extend(
        [
            "",
            "## Scenario Averages",
            "",
            "| Scenario | Best method | Best accuracy | Uniform acc. | Mean accuracy | Mean help vs uniform | Mean harm vs uniform |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in grouped:
        lines.append(
            "| {scenario} | {best_method} | {best_accuracy} | {uniform_accuracy} | {mean_accuracy} | {mean_help_rate_vs_uniform} | {mean_harm_rate_vs_uniform} |".format(
                scenario=row["scenario"],
                best_method=row["best_method"],
                best_accuracy=fmt(row["best_accuracy"]),
                uniform_accuracy=fmt(row["uniform_accuracy"]),
                mean_accuracy=fmt(row["mean_accuracy"]),
                mean_help_rate_vs_uniform=fmt(row["mean_help_rate_vs_uniform"]),
                mean_harm_rate_vs_uniform=fmt(row["mean_harm_rate_vs_uniform"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy sensitivity-aware budget bridge.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument("--output-md", help="Optional Markdown summary.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-examples", type=int, default=96)
    parser.add_argument("--test-examples", type=int, default=192)
    parser.add_argument("--slots", type=int, default=12)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--protected-slots", type=int, default=3)
    parser.add_argument("--protected-channels", type=int, default=4)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--signal-scale", type=float, default=2.25)
    parser.add_argument("--noise-scale", type=float, default=0.14)
    parser.add_argument("--outlier-scale", type=float, default=7.5)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("aligned", "rotated", "outlier", "slot_permuted"),
        default=["aligned", "rotated", "outlier", "slot_permuted"],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToySensitivityBudgetConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        slots=args.slots,
        channels=args.channels,
        classes=args.classes,
        protected_slots=args.protected_slots,
        protected_channels=args.protected_channels,
        bits=args.bits,
        signal_scale=args.signal_scale,
        noise_scale=args.noise_scale,
        outlier_scale=args.outlier_scale,
        scenarios=tuple(args.scenarios),
    )
    rows = run_experiment(config)
    payload = {"config": json.loads(json.dumps(asdict(config))), "rows": rows}
    output = pathlib.Path(args.output)
    write_jsonl(rows, output)
    if args.output_md:
        write_markdown_summary(rows, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
