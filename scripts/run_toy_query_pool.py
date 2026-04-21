#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyConfig:
    seed: int = 0
    train_examples: int = 384
    test_examples: int = 192
    dim: int = 24
    slots: int = 32
    classes: int = 5
    top_k: int = 4
    pool_slots: int = 4
    route_atoms: int = 4
    epochs: int = 120
    lr: float = 2e-2
    rec_weight: float = 0.2
    noise: float = 0.08


@dataclass
class ToyBatch:
    q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor


def _orthogonal(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r)).clamp(min=-1.0, max=1.0)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _make_base_tensors(config: ToyConfig) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(config.seed)
    dim = config.dim
    slots = config.slots
    return {
        "slot_k": torch.randn(slots, dim, dim, generator=gen) / math.sqrt(dim),
        "slot_v": torch.randn(slots, dim, dim, generator=gen) / math.sqrt(dim),
        "query": torch.randn(dim, dim, generator=gen) / math.sqrt(dim),
        "classifier": torch.randn(dim, config.classes, generator=gen) / math.sqrt(dim),
        "rotation": _orthogonal(dim, gen),
        "slot_perm": torch.randperm(slots, generator=gen),
    }


def make_toy_batch(
    config: ToyConfig,
    *,
    scenario: str,
    split: str,
    base: dict[str, torch.Tensor] | None = None,
) -> ToyBatch:
    if scenario not in {"aligned", "rotated", "outlier", "slot_permuted"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    base = _make_base_tensors(config) if base is None else base
    offset = 10_000 if split == "test" else 0
    gen = torch.Generator().manual_seed(config.seed + offset + sum(ord(ch) for ch in scenario))
    count = config.train_examples if split == "train" else config.test_examples
    z = torch.randn(count, config.dim, generator=gen)
    q = z @ base["query"]
    K = torch.einsum("nd,sde->nse", z, base["slot_k"])
    V = torch.einsum("nd,sde->nse", z, base["slot_v"])
    K = K + config.noise * torch.randn(K.shape, generator=gen)
    V = V + config.noise * torch.randn(V.shape, generator=gen)

    if scenario == "rotated":
        K = K @ base["rotation"]
        V = V @ base["rotation"]
    elif scenario == "outlier":
        scale = torch.ones(config.dim)
        scale[: max(1, config.dim // 8)] = 8.0
        K = K * scale.view(1, 1, -1)
        V = V * scale.flip(0).view(1, 1, -1)
    elif scenario == "slot_permuted":
        K = K[:, base["slot_perm"]]
        V = V[:, base["slot_perm"]]

    y = (z @ base["classifier"]).argmax(dim=-1)
    return ToyBatch(q=q.float(), K=K.float(), V=V.float(), z=z.float(), y=y.long())


def _route_metrics(weights: torch.Tensor, *, margin_k: int = 1) -> dict[str, float]:
    probs = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    winners = probs.argmax(dim=-1)
    counts = torch.bincount(winners, minlength=probs.shape[-1]).float()
    collision_rate = counts.max() / max(int(winners.numel()), 1)
    dead_slot_rate = (counts == 0).float().mean()
    sorted_probs = probs.sort(dim=-1, descending=True).values
    idx = min(max(int(margin_k), 1), sorted_probs.shape[-1] - 1)
    margin = (sorted_probs[:, 0] - sorted_probs[:, idx]).mean()
    return {
        "route_entropy": float(entropy.item()),
        "slot_collision_rate": float(collision_rate.item()),
        "dead_slot_rate": float(dead_slot_rate.item()),
        "top_margin": float(margin.item()),
    }


def _atom_metrics(atom_weights: torch.Tensor) -> dict[str, float]:
    probs = atom_weights / atom_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropy = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    winners = probs.argmax(dim=-1)
    counts = torch.bincount(winners, minlength=probs.shape[-1]).float()
    collision_rate = counts.max() / max(int(winners.numel()), 1)
    dead_atom_rate = (counts == 0).float().mean()
    sorted_probs = probs.sort(dim=-1, descending=True).values
    margin = (sorted_probs[:, 0] - sorted_probs[:, min(1, sorted_probs.shape[-1] - 1)]).mean()
    return {
        "atom_entropy": float(entropy.item()),
        "atom_collision_rate": float(collision_rate.item()),
        "dead_atom_rate": float(dead_atom_rate.item()),
        "atom_top_margin": float(margin.item()),
    }


def _precondition_metrics(preconditioned: torch.Tensor, original: torch.Tensor) -> dict[str, float]:
    cosine = F.cosine_similarity(preconditioned, original, dim=-1)
    norm_ratio = preconditioned.norm(dim=-1) / original.norm(dim=-1).clamp_min(1e-8)
    abs_scale_ratio = preconditioned.abs().mean(dim=-1) / original.abs().mean(dim=-1).clamp_min(1e-8)
    condition_proxy = preconditioned.abs().amax(dim=-1) / preconditioned.abs().amin(dim=-1).clamp_min(1e-8)
    return {
        "precondition_condition_proxy": float(condition_proxy.mean().item()),
        "precondition_cosine_drift": float(cosine.mean().item()),
        "precondition_norm_ratio": float(norm_ratio.mean().item()),
        "precondition_abs_scale_ratio": float(abs_scale_ratio.mean().item()),
    }


class TopKReadout(nn.Module):
    def __init__(self, dim: int, classes: int, top_k: int) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum("nd,nsd->ns", batch.q, batch.K) / math.sqrt(batch.q.shape[-1])
        keep = min(max(self.top_k, 1), logits.shape[-1])
        top = torch.topk(logits, k=keep, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(dim=-1, index=top.indices, src=top.values)
        weights = torch.softmax(masked, dim=-1)
        summary = torch.einsum("ns,nsd->nd", weights, batch.V)
        return summary, weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


class QueryPoolReadout(nn.Module):
    def __init__(self, dim: int, classes: int, pool_slots: int) -> None:
        super().__init__()
        self.pool_queries = nn.Parameter(torch.randn(pool_slots, dim) / math.sqrt(dim))
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)
        self.last_atom_weights: torch.Tensor | None = None

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.query_proj(batch.q)
        beta = torch.softmax(q_proj @ self.pool_queries.T / math.sqrt(batch.q.shape[-1]), dim=-1)
        pool_q = q_proj[:, None, :] + self.pool_queries[None, :, :]
        slot_logits = torch.einsum("npd,nsd->nps", pool_q, batch.K) / math.sqrt(batch.q.shape[-1])
        omega = torch.softmax(slot_logits, dim=-1)
        pool_values = torch.einsum("nps,nsd->npd", omega, batch.V)
        summary = torch.einsum("np,npd->nd", beta, pool_values)
        effective_weights = torch.einsum("np,nps->ns", beta, omega)
        self.last_atom_weights = beta.detach()
        return summary, effective_weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


class PreconditionedQueryPoolReadout(nn.Module):
    def __init__(self, dim: int, classes: int, pool_slots: int) -> None:
        super().__init__()
        self.pool_queries = nn.Parameter(torch.randn(pool_slots, dim) / math.sqrt(dim))
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.precondition_log_scale = nn.Parameter(torch.zeros(dim))
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)
        self.last_atom_weights: torch.Tensor | None = None
        self.last_preconditioned_q: torch.Tensor | None = None
        self.last_original_q: torch.Tensor | None = None

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.query_proj(batch.q)
        scale = torch.exp(self.precondition_log_scale).clamp(1e-3, 1e3)
        q_pre = q_proj * scale
        beta = torch.softmax(q_pre @ self.pool_queries.T / math.sqrt(batch.q.shape[-1]), dim=-1)
        pool_q = q_pre[:, None, :] + self.pool_queries[None, :, :]
        slot_logits = torch.einsum("npd,nsd->nps", pool_q, batch.K) / math.sqrt(batch.q.shape[-1])
        omega = torch.softmax(slot_logits, dim=-1)
        pool_values = torch.einsum("nps,nsd->npd", omega, batch.V)
        summary = torch.einsum("np,npd->nd", beta, pool_values)
        effective_weights = torch.einsum("np,nps->ns", beta, omega)
        self.last_atom_weights = beta.detach()
        self.last_preconditioned_q = q_pre.detach()
        self.last_original_q = q_proj.detach()
        return summary, effective_weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


class RouteAtomReadout(nn.Module):
    def __init__(self, dim: int, classes: int, route_atoms: int) -> None:
        super().__init__()
        self.route_atoms = int(route_atoms)
        self.route_bank = nn.Parameter(torch.randn(self.route_atoms, dim) / math.sqrt(dim))
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)
        self.last_atom_weights: torch.Tensor | None = None

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.query_proj(batch.q)
        atom_logits = q_proj @ self.route_bank.T / math.sqrt(batch.q.shape[-1])
        atom_weights = torch.softmax(atom_logits, dim=-1)
        atom_q = q_proj[:, None, :] * torch.tanh(self.route_bank)[None, :, :]
        slot_logits = torch.einsum("nad,nsd->nas", atom_q, batch.K) / math.sqrt(batch.q.shape[-1])
        slot_weights = torch.softmax(slot_logits, dim=-1)
        atom_values = torch.einsum("nas,nsd->nad", slot_weights, batch.V)
        summary = torch.einsum("na,nad->nd", atom_weights, atom_values)
        effective_weights = torch.einsum("na,nas->ns", atom_weights, slot_weights)
        self.last_atom_weights = atom_weights.detach()
        return summary, effective_weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


def _train_model(model: nn.Module, train: ToyBatch, config: ToyConfig) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits, recon, _ = model(train)
        loss = F.cross_entropy(logits, train.y) + config.rec_weight * F.mse_loss(recon, train.z)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _evaluate_model(model: nn.Module, batch: ToyBatch, *, margin_k: int) -> dict[str, float]:
    model.eval()
    logits, recon, weights = model(batch)
    pred = logits.argmax(dim=-1)
    metrics = {
        "task_acc": float((pred == batch.y).float().mean().item()),
        "rec_mse": float(F.mse_loss(recon, batch.z).item()),
    }
    metrics.update(_route_metrics(weights, margin_k=margin_k))
    atom_weights = getattr(model, "last_atom_weights", None)
    if atom_weights is None:
        metrics.update(
            {
                "atom_entropy": None,
                "atom_collision_rate": None,
                "dead_atom_rate": None,
                "atom_top_margin": None,
            }
        )
    else:
        metrics.update(_atom_metrics(atom_weights))
    preconditioned_q = getattr(model, "last_preconditioned_q", None)
    original_q = getattr(model, "last_original_q", None)
    if preconditioned_q is None or original_q is None:
        metrics.update(
            {
                "precondition_condition_proxy": None,
                "precondition_cosine_drift": None,
                "precondition_norm_ratio": None,
                "precondition_abs_scale_ratio": None,
            }
        )
    else:
        metrics.update(_precondition_metrics(preconditioned_q, original_q))
    return metrics


def run_experiment(config: ToyConfig, scenarios: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = _make_base_tensors(config)
    method_seed_offsets = {
        "topk": 101,
        "query_pool": 211,
        "preconditioned_query_pool": 307,
        "route_atom": 401,
    }
    for scenario_idx, scenario in enumerate(scenarios):
        train = make_toy_batch(config, scenario=scenario, split="train", base=base)
        test = make_toy_batch(config, scenario=scenario, split="test", base=base)
        methods: list[tuple[str, Any, int]] = [
            ("topk", lambda: TopKReadout(config.dim, config.classes, config.top_k), config.top_k),
            ("query_pool", lambda: QueryPoolReadout(config.dim, config.classes, config.pool_slots), 1),
            (
                "preconditioned_query_pool",
                lambda: PreconditionedQueryPoolReadout(config.dim, config.classes, config.pool_slots),
                1,
            ),
            ("route_atom", lambda: RouteAtomReadout(config.dim, config.classes, config.route_atoms), 1),
        ]
        for method, factory, margin_k in methods:
            torch.manual_seed(config.seed + scenario_idx * 10_000 + method_seed_offsets[method])
            model = factory()
            _train_model(model, train, config)
            metrics = _evaluate_model(model, test, margin_k=margin_k)
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "budget": (
                        config.top_k
                        if method == "topk"
                        else config.pool_slots
                        if method in {"query_pool", "preconditioned_query_pool"}
                        else config.route_atoms
                    ),
                    **metrics,
                }
            )
    return rows


def write_markdown_summary(rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    lines = [
        "# Toy Query-Pool Benchmark",
        "",
        "| Scenario | Method | Budget | Task acc | Rec MSE | Route entropy | Collision | Dead slots | Top margin | Atom entropy | Atom collision | Dead atoms | Atom margin | Precond cond. | Cosine drift | Norm ratio | Abs scale ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {method} | {budget} | {task_acc} | {rec_mse} | "
            "{route_entropy} | {slot_collision_rate} | {dead_slot_rate} | {top_margin} | "
            "{atom_entropy} | {atom_collision_rate} | {dead_atom_rate} | {atom_top_margin} |".format(
                scenario=row["scenario"],
                method=row["method"],
                budget=row["budget"],
                task_acc=fmt(row["task_acc"]),
                rec_mse=fmt(row["rec_mse"]),
                route_entropy=fmt(row["route_entropy"]),
                slot_collision_rate=fmt(row["slot_collision_rate"]),
                dead_slot_rate=fmt(row["dead_slot_rate"]),
                top_margin=fmt(row["top_margin"]),
                atom_entropy=fmt(row["atom_entropy"]),
                atom_collision_rate=fmt(row["atom_collision_rate"]),
                dead_atom_rate=fmt(row["dead_atom_rate"]),
                atom_top_margin=fmt(row["atom_top_margin"]),
                precondition_condition_proxy=fmt(row["precondition_condition_proxy"]),
                precondition_cosine_drift=fmt(row["precondition_cosine_drift"]),
                precondition_norm_ratio=fmt(row["precondition_norm_ratio"]),
                precondition_abs_scale_ratio=fmt(row["precondition_abs_scale_ratio"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic top-k vs query-pool KV transport benchmark.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=384)
    parser.add_argument("--test-examples", type=int, default=192)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--pool-slots", type=int, default=4)
    parser.add_argument("--route-atoms", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--rec-weight", type=float, default=0.2)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["aligned", "rotated", "outlier", "slot_permuted"],
        choices=["aligned", "rotated", "outlier", "slot_permuted"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ToyConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        slots=args.slots,
        classes=args.classes,
        top_k=args.top_k,
        pool_slots=args.pool_slots,
        route_atoms=args.route_atoms,
        epochs=args.epochs,
        lr=args.lr,
        rec_weight=args.rec_weight,
        noise=args.noise,
    )
    rows = run_experiment(config, list(args.scenarios))
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(config), "rows": rows}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        write_markdown_summary(rows, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
