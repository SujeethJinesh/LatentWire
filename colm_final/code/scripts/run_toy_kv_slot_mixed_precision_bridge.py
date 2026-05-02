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
class ToyKVSlotMixedPrecisionConfig:
    seed: int = 0
    train_examples: int = 192
    test_examples: int = 128
    slots: int = 16
    dim: int = 32
    classes: int = 6
    low_bits: int = 3
    key_high_bits: int = 6
    value_high_bits: int = 5
    protected_key_channels: int = 2
    protected_value_channels: int = 2
    route_signal_channels: int = 4
    answer_signal_channels: int = 4
    route_signal_scale: float = 4.5
    answer_signal_scale: float = 4.0
    key_noise: float = 0.18
    value_noise: float = 0.28
    query_noise: float = 0.22


@dataclass(frozen=True)
class ToyBatch:
    q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    route_target: torch.Tensor
    answer_target: torch.Tensor


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
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, str]:
    if dim > 0 and dim & (dim - 1) == 0:
        basis = _normalized_hadamard(dim, device=device, dtype=dtype)
        sign_gen = _make_generator(seed + 31)
        signs = torch.where(
            torch.randn(dim, generator=sign_gen, device=device) >= 0,
            torch.ones(dim, device=device, dtype=dtype),
            -torch.ones(dim, device=device, dtype=dtype),
        )
        perm = torch.randperm(dim, generator=_make_generator(seed + 53), device=device)
        return basis @ torch.diag(signs)[:, perm], "hadamard"
    return _orthogonal_matrix(dim, _make_generator(seed + 17)).to(device=device, dtype=dtype), "orthogonal"


def _apply_rotation(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return x @ matrix


def _inverse_rotation(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return x @ matrix.T


def _bytes_for_values(count: int, bits: int) -> int:
    return int(math.ceil(count * bits / 8.0))


def _estimate_uniform_bytes(num_examples: int, slots: int, dim: int, bits: int) -> int:
    return 2 * _bytes_for_values(num_examples * slots * dim, bits)


def _estimate_tensor_bytes(num_examples: int, slots: int, dim: int, bits: int) -> int:
    return _bytes_for_values(num_examples * slots * dim, bits)


def _estimate_protected_bytes(
    num_examples: int,
    slots: int,
    dim: int,
    bits: int,
    protected_bits: int,
    protected_channels: int,
) -> int:
    protected = max(1, min(int(protected_channels), max(dim - 1, 1)))
    bulk = dim - protected
    index_bytes_per_channel = max(1, math.ceil(math.log2(max(dim, 2)) / 8.0))
    bulk_bytes = _bytes_for_values(num_examples * slots * bulk, bits)
    protected_bytes = _bytes_for_values(num_examples * slots * protected, protected_bits)
    metadata = protected * index_bytes_per_channel + 4
    return bulk_bytes + protected_bytes + metadata


def _estimate_mixed_bytes(
    num_examples: int,
    slots: int,
    dim: int,
    low_bits: int,
    high_bits: int,
    protected_channels: int,
) -> int:
    protected = max(1, min(int(protected_channels), max(dim - 1, 1)))
    bulk = dim - protected
    index_bytes_per_channel = max(1, math.ceil(math.log2(max(dim, 2)) / 8.0))
    bulk_bytes = _bytes_for_values(num_examples * slots * bulk, low_bits)
    protected_bytes = _bytes_for_values(num_examples * slots * protected, high_bits)
    metadata = protected * index_bytes_per_channel + 4
    return bulk_bytes + protected_bytes + metadata


def _select_salient_channels(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("Expected a tensor with a feature dimension")
    reduce_dims = tuple(range(x.ndim - 1))
    energy = x.pow(2).mean(dim=reduce_dims)
    k = max(1, min(int(k), x.shape[-1]))
    return torch.topk(energy, k=k, largest=True).indices.sort().values


def _symmetric_quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(x / scale).clamp(-qmax, qmax)
    return codes * scale


def _quantize_channels(
    x: torch.Tensor,
    *,
    low_bits: int,
    protected_idx: torch.Tensor | None = None,
    protected_bits: int | None = None,
) -> torch.Tensor:
    recon = x.clone()
    if protected_idx is None or protected_idx.numel() == 0:
        return _symmetric_quantize(recon, low_bits)
    protected_idx = protected_idx.sort().values
    mask = torch.ones(x.shape[-1], dtype=torch.bool, device=x.device)
    mask[protected_idx] = False
    if mask.any():
        recon[..., mask] = _symmetric_quantize(x[..., mask], low_bits)
    if protected_bits is not None and protected_idx.numel() > 0:
        recon[..., protected_idx] = _symmetric_quantize(x[..., protected_idx], protected_bits)
    return recon


def _make_base_components(config: ToyKVSlotMixedPrecisionConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    slot_labels = torch.randperm(config.slots, generator=gen) % config.classes

    key_templates = torch.randn(config.slots, config.dim, generator=gen, dtype=torch.float32)
    key_templates[:, : max(1, config.route_signal_channels)] *= float(config.route_signal_scale)

    class_templates = torch.randn(config.classes, config.dim, generator=gen, dtype=torch.float32)
    class_templates[:, -max(1, config.answer_signal_channels) :] *= float(config.answer_signal_scale)
    value_templates = class_templates[slot_labels]
    return {
        "key_templates": key_templates,
        "class_templates": class_templates,
        "value_templates": value_templates,
        "slot_labels": slot_labels,
    }


def _make_split(
    config: ToyKVSlotMixedPrecisionConfig,
    *,
    split: str,
    base: dict[str, torch.Tensor],
) -> ToyBatch:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    count = config.train_examples if split == "train" else config.test_examples
    offset = 11_000 if split == "test" else 0
    gen = _make_generator(config.seed + offset)

    route_target = torch.randperm(count, generator=gen) % config.slots
    q = base["key_templates"][route_target] + float(config.query_noise) * torch.randn(
        count, config.dim, generator=gen, dtype=torch.float32
    )

    K = base["key_templates"].unsqueeze(0).expand(count, -1, -1).clone()
    K = K + float(config.key_noise) * torch.randn(count, config.slots, config.dim, generator=gen)

    V = base["value_templates"].unsqueeze(0).expand(count, -1, -1).clone()
    V = V + float(config.value_noise) * torch.randn(count, config.slots, config.dim, generator=gen)

    answer_target = base["slot_labels"][route_target]
    return ToyBatch(
        q=q.float(),
        K=K.float(),
        V=V.float(),
        route_target=route_target.long(),
        answer_target=answer_target.long(),
    )


def _route_answer_metrics(
    q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    class_templates: torch.Tensor,
    route_target: torch.Tensor,
    answer_target: torch.Tensor,
) -> tuple[float, float]:
    route_scores = torch.einsum("nd,nsd->ns", q, K)
    route_pred = route_scores.argmax(dim=-1)
    route_accuracy = float((route_pred == route_target).float().mean().item())

    selected_values = V[torch.arange(V.shape[0], device=V.device), route_pred]
    answer_scores = selected_values @ class_templates.T
    answer_pred = answer_scores.argmax(dim=-1)
    answer_accuracy = float((answer_pred == answer_target).float().mean().item())
    return route_accuracy, answer_accuracy


def _evaluate_row(
    *,
    method: str,
    basis: str,
    config: ToyKVSlotMixedPrecisionConfig,
    test_batch: ToyBatch,
    class_templates: torch.Tensor,
    key_recon: torch.Tensor,
    value_recon: torch.Tensor,
    key_selected_indices: torch.Tensor,
    value_selected_indices: torch.Tensor,
    bytes_estimate: float,
) -> dict[str, Any]:
    route_accuracy, answer_accuracy = _route_answer_metrics(
        test_batch.q,
        key_recon,
        value_recon,
        class_templates,
        test_batch.route_target,
        test_batch.answer_target,
    )
    return {
        "method": method,
        "basis": basis,
        "seed": int(config.seed),
        "slots": int(config.slots),
        "dim": int(config.dim),
        "classes": int(config.classes),
        "low_bits": int(config.low_bits),
        "key_high_bits": int(config.key_high_bits),
        "value_high_bits": int(config.value_high_bits),
        "protected_key_channels": int(config.protected_key_channels),
        "protected_value_channels": int(config.protected_value_channels),
        "route_accuracy": route_accuracy,
        "answer_accuracy": answer_accuracy,
        "key_mse": float(F.mse_loss(key_recon, test_batch.K).item()),
        "value_mse": float(F.mse_loss(value_recon, test_batch.V).item()),
        "bytes_estimate": float(bytes_estimate),
        "key_selected_indices": [int(i) for i in key_selected_indices.tolist()],
        "value_selected_indices": [int(i) for i in value_selected_indices.tolist()],
        "basis_orthogonality_error": 0.0,
    }


def run_experiment(config: ToyKVSlotMixedPrecisionConfig) -> dict[str, Any]:
    base = _make_base_components(config)
    train_batch = _make_split(config, split="train", base=base)
    test_batch = _make_split(config, split="test", base=base)

    key_selected = _select_salient_channels(train_batch.K, config.protected_key_channels)
    value_selected = _select_salient_channels(train_batch.V, config.protected_value_channels)

    rows: list[dict[str, Any]] = []

    uniform_key = _quantize_channels(test_batch.K, low_bits=config.low_bits)
    uniform_value = _quantize_channels(test_batch.V, low_bits=config.low_bits)
    uniform_bytes = float(_estimate_uniform_bytes(config.test_examples, config.slots, config.dim, config.low_bits))
    uniform_row = _evaluate_row(
        method="uniform_low_bit",
        basis="none",
        config=config,
        test_batch=test_batch,
        class_templates=base["class_templates"],
        key_recon=uniform_key,
        value_recon=uniform_value,
        key_selected_indices=torch.empty(0, dtype=torch.long),
        value_selected_indices=torch.empty(0, dtype=torch.long),
        bytes_estimate=uniform_bytes,
    )
    rows.append(uniform_row)

    key_protected_key = _quantize_channels(
        test_batch.K,
        low_bits=config.low_bits,
        protected_idx=key_selected,
        protected_bits=config.key_high_bits,
    )
    key_protected_value = _quantize_channels(test_batch.V, low_bits=config.low_bits)
    key_protected_bytes = float(
        _estimate_protected_bytes(
            config.test_examples,
            config.slots,
            config.dim,
            config.low_bits,
            config.key_high_bits,
            config.protected_key_channels,
        )
        + _estimate_tensor_bytes(config.test_examples, config.slots, config.dim, config.low_bits)
    )
    key_protected_row = _evaluate_row(
        method="key_protected_value_low",
        basis="none",
        config=config,
        test_batch=test_batch,
        class_templates=base["class_templates"],
        key_recon=key_protected_key,
        value_recon=key_protected_value,
        key_selected_indices=key_selected,
        value_selected_indices=torch.empty(0, dtype=torch.long),
        bytes_estimate=key_protected_bytes,
    )
    rows.append(key_protected_row)

    value_protected_key = _quantize_channels(test_batch.K, low_bits=config.low_bits)
    value_protected_value = _quantize_channels(
        test_batch.V,
        low_bits=config.low_bits,
        protected_idx=value_selected,
        protected_bits=config.value_high_bits,
    )
    value_protected_bytes = float(
        _estimate_tensor_bytes(config.test_examples, config.slots, config.dim, config.low_bits)
        + _estimate_protected_bytes(
            config.test_examples,
            config.slots,
            config.dim,
            config.low_bits,
            config.value_high_bits,
            config.protected_value_channels,
        )
    )
    value_protected_row = _evaluate_row(
        method="value_protected_key_low",
        basis="none",
        config=config,
        test_batch=test_batch,
        class_templates=base["class_templates"],
        key_recon=value_protected_key,
        value_recon=value_protected_value,
        key_selected_indices=torch.empty(0, dtype=torch.long),
        value_selected_indices=value_selected,
        bytes_estimate=value_protected_bytes,
    )
    rows.append(value_protected_row)

    mixed_key = _quantize_channels(
        test_batch.K,
        low_bits=config.low_bits,
        protected_idx=key_selected,
        protected_bits=config.key_high_bits,
    )
    mixed_value = _quantize_channels(
        test_batch.V,
        low_bits=config.low_bits,
        protected_idx=value_selected,
        protected_bits=config.value_high_bits,
    )
    mixed_bytes = float(
        _estimate_mixed_bytes(
            config.test_examples,
            config.slots,
            config.dim,
            config.low_bits,
            config.key_high_bits,
            config.protected_key_channels,
        )
        + _estimate_mixed_bytes(
            config.test_examples,
            config.slots,
            config.dim,
            config.low_bits,
            config.value_high_bits,
            config.protected_value_channels,
        )
    )
    mixed_row = _evaluate_row(
        method="mixed_kv_precision",
        basis="none",
        config=config,
        test_batch=test_batch,
        class_templates=base["class_templates"],
        key_recon=mixed_key,
        value_recon=mixed_value,
        key_selected_indices=key_selected,
        value_selected_indices=value_selected,
        bytes_estimate=mixed_bytes,
    )
    rows.append(mixed_row)

    rotation, basis = _rotation_matrix(config.dim, seed=config.seed + 313, device=test_batch.K.device, dtype=test_batch.K.dtype)
    rotated_key = _apply_rotation(test_batch.K, rotation)
    rotated_value = _apply_rotation(test_batch.V, rotation)
    rotated_key_recon = _inverse_rotation(_quantize_channels(rotated_key, low_bits=config.low_bits), rotation)
    rotated_value_recon = _inverse_rotation(_quantize_channels(rotated_value, low_bits=config.low_bits), rotation)
    rotated_row = _evaluate_row(
        method="incoherent_basis_rotation",
        basis=basis,
        config=config,
        test_batch=test_batch,
        class_templates=base["class_templates"],
        key_recon=rotated_key_recon,
        value_recon=rotated_value_recon,
        key_selected_indices=torch.empty(0, dtype=torch.long),
        value_selected_indices=torch.empty(0, dtype=torch.long),
        bytes_estimate=uniform_bytes,
    )
    rotated_row["basis_orthogonality_error"] = float(
        torch.linalg.norm(rotation.T @ rotation - torch.eye(config.dim, dtype=rotation.dtype)).item()
    )
    rows.append(rotated_row)

    uniform_route = uniform_row["route_accuracy"]
    uniform_answer = uniform_row["answer_accuracy"]
    for row in rows[1:]:
        route_delta = row["route_accuracy"] - uniform_route
        answer_delta = row["answer_accuracy"] - uniform_answer
        row["route_help_vs_uniform"] = float(max(0.0, route_delta))
        row["route_harm_vs_uniform"] = float(max(0.0, -route_delta))
        row["answer_help_vs_uniform"] = float(max(0.0, answer_delta))
        row["answer_harm_vs_uniform"] = float(max(0.0, -answer_delta))
        row["route_delta_vs_uniform"] = float(route_delta)
        row["answer_delta_vs_uniform"] = float(answer_delta)
    uniform_row["route_help_vs_uniform"] = 0.0
    uniform_row["route_harm_vs_uniform"] = 0.0
    uniform_row["answer_help_vs_uniform"] = 0.0
    uniform_row["answer_harm_vs_uniform"] = 0.0
    uniform_row["route_delta_vs_uniform"] = 0.0
    uniform_row["answer_delta_vs_uniform"] = 0.0
    uniform_row["basis_orthogonality_error"] = 0.0

    return {"config": asdict(config), "rows": rows}


def write_json(payload: dict[str, Any], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    lines = [
        "# Toy K/V Slot Mixed-Precision Bridge",
        "",
        "This toy separates key transport from value transport.",
        "Keys are responsible for route retrieval, values are responsible for answer content,",
        "and an incoherent basis rotation probes whether low-bit quantization is sensitive to",
        "coordinate alignment before reconstruction.",
        "",
        "| Method | Basis | Route acc | Answer acc | K MSE | V MSE | Bytes estimate | Route help | Answer help |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {basis} | {route_accuracy:.4f} | {answer_accuracy:.4f} | {key_mse:.4f} | {value_mse:.4f} | {bytes_estimate:.1f} | {route_help_vs_uniform:.4f} | {answer_help_vs_uniform:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "The key-protected variant should primarily help route recovery, the value-protected",
            "variant should primarily help answer reconstruction, the mixed K/V allocation should",
            "offer the best combined tradeoff, and the basis rotation should change the quantization",
            "geometry without changing the underlying task semantics.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy K/V slot mixed-precision bridge.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=192)
    parser.add_argument("--test-examples", type=int, default=128)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--low-bits", type=int, default=3)
    parser.add_argument("--key-high-bits", type=int, default=6)
    parser.add_argument("--value-high-bits", type=int, default=5)
    parser.add_argument("--protected-key-channels", type=int, default=2)
    parser.add_argument("--protected-value-channels", type=int, default=2)
    parser.add_argument("--route-signal-channels", type=int, default=4)
    parser.add_argument("--answer-signal-channels", type=int, default=4)
    parser.add_argument("--route-signal-scale", type=float, default=4.5)
    parser.add_argument("--answer-signal-scale", type=float, default=4.0)
    parser.add_argument("--key-noise", type=float, default=0.18)
    parser.add_argument("--value-noise", type=float, default=0.28)
    parser.add_argument("--query-noise", type=float, default=0.22)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyKVSlotMixedPrecisionConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        slots=args.slots,
        dim=args.dim,
        classes=args.classes,
        low_bits=args.low_bits,
        key_high_bits=args.key_high_bits,
        value_high_bits=args.value_high_bits,
        protected_key_channels=args.protected_key_channels,
        protected_value_channels=args.protected_value_channels,
        route_signal_channels=args.route_signal_channels,
        answer_signal_channels=args.answer_signal_channels,
        route_signal_scale=args.route_signal_scale,
        answer_signal_scale=args.answer_signal_scale,
        key_noise=args.key_noise,
        value_noise=args.value_noise,
        query_noise=args.query_noise,
    )
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    write_json(payload, output)
    write_markdown_summary(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
