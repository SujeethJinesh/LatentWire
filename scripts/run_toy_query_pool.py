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
    kv_route_budget: int = 2
    kv_value_budget: int = 4
    codebook_size: int = 4
    residual_codebook_size: int = 4
    protected_channels: int = 2
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


def _kv_asymmetry_metrics(
    route_weights: torch.Tensor,
    value_weights: torch.Tensor,
    route_indices: torch.Tensor,
    value_indices: torch.Tensor,
    route_summary: torch.Tensor,
    value_summary: torch.Tensor,
    gate: torch.Tensor,
) -> dict[str, float]:
    route_probs = route_weights / route_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    value_probs = value_weights / value_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    route_entropy = (-(route_probs * route_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    value_entropy = (-(value_probs * value_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()

    route_sorted = route_probs.sort(dim=-1, descending=True).values
    value_sorted = value_probs.sort(dim=-1, descending=True).values
    route_margin = (route_sorted[:, 0] - route_sorted[:, min(1, route_sorted.shape[-1] - 1)]).mean()
    value_margin = (value_sorted[:, 0] - value_sorted[:, min(1, value_sorted.shape[-1] - 1)]).mean()

    route_winners = route_probs.argmax(dim=-1)
    value_winners = value_probs.argmax(dim=-1)
    route_counts = torch.bincount(route_winners, minlength=route_probs.shape[-1]).float()
    value_counts = torch.bincount(value_winners, minlength=value_probs.shape[-1]).float()
    route_collision_rate = route_counts.max() / max(int(route_winners.numel()), 1)
    value_collision_rate = value_counts.max() / max(int(value_winners.numel()), 1)
    route_dead_rate = (route_counts == 0).float().mean()
    value_dead_rate = (value_counts == 0).float().mean()

    route_keep = route_indices.shape[-1]
    value_keep = value_indices.shape[-1]
    overlaps = []
    jaccards = []
    for route_idx, value_idx in zip(route_indices, value_indices):
        route_set = set(route_idx.tolist())
        value_set = set(value_idx.tolist())
        intersection = len(route_set & value_set)
        union = len(route_set | value_set)
        overlaps.append(intersection / max(min(len(route_set), len(value_set)), 1))
        jaccards.append(intersection / max(union, 1))

    kl = F.kl_div(route_probs.clamp_min(1e-8).log(), value_probs.clamp_min(1e-8), reduction="none").sum(dim=-1).mean()
    cosine = F.cosine_similarity(route_summary, value_summary, dim=-1).mean()
    gap = F.mse_loss(route_summary, value_summary)

    return {
        "kv_route_entropy": float(route_entropy.item()),
        "kv_value_entropy": float(value_entropy.item()),
        "kv_route_collision_rate": float(route_collision_rate.item()),
        "kv_value_collision_rate": float(value_collision_rate.item()),
        "kv_route_dead_rate": float(route_dead_rate.item()),
        "kv_value_dead_rate": float(value_dead_rate.item()),
        "kv_route_top_margin": float(route_margin.item()),
        "kv_value_top_margin": float(value_margin.item()),
        "kv_route_keep_fraction": float(route_keep / route_probs.shape[-1]),
        "kv_value_keep_fraction": float(value_keep / value_probs.shape[-1]),
        "kv_route_value_overlap": float(sum(overlaps) / max(len(overlaps), 1)),
        "kv_route_value_jaccard": float(sum(jaccards) / max(len(jaccards), 1)),
        "kv_route_value_kl": float(kl.item()),
        "kv_route_value_cosine": float(cosine.item()),
        "kv_route_value_gap": float(gap.item()),
        "kv_gate_mean": float(gate.mean().item()),
        "kv_gate_std": float(gate.std(unbiased=False).item()),
    }


def _codebook_metrics(
    code_weights: torch.Tensor,
    slot_code_weights: torch.Tensor,
    codebook: torch.Tensor,
    query_proj: torch.Tensor,
    slot_keys: torch.Tensor,
    *,
    prefix: str = "",
) -> dict[str, float]:
    code_probs = code_weights / code_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    slot_probs = slot_code_weights / slot_code_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    code_entropy = (-(code_probs * code_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()
    slot_entropy = (-(slot_probs * slot_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean()

    code_winners = code_probs.argmax(dim=-1)
    slot_winners = slot_probs.argmax(dim=-1)
    code_counts = torch.bincount(code_winners, minlength=code_probs.shape[-1]).float()
    slot_counts = torch.bincount(slot_winners.view(-1), minlength=slot_probs.shape[-1]).float()

    code_collision_rate = code_counts.max() / max(int(code_winners.numel()), 1)
    slot_collision_rate = slot_counts.max() / max(int(slot_winners.numel()), 1)
    dead_code_rate = (code_counts == 0).float().mean()
    dead_slot_code_rate = (slot_counts == 0).float().mean()
    code_sorted = code_probs.sort(dim=-1, descending=True).values
    slot_sorted = slot_probs.sort(dim=-1, descending=True).values
    code_margin = (code_sorted[:, 0] - code_sorted[:, min(1, code_sorted.shape[-1] - 1)]).mean()
    slot_margin = (slot_sorted[:, :, 0] - slot_sorted[:, :, min(1, slot_sorted.shape[-1] - 1)]).mean()

    code_recon = code_probs @ codebook
    slot_recon = torch.einsum("nsc,cd->nsd", slot_probs, codebook)
    query_recon_mse = F.mse_loss(code_recon, query_proj)
    slot_recon_mse = F.mse_loss(slot_recon, slot_keys)
    query_recon_cosine = F.cosine_similarity(code_recon, query_proj, dim=-1).mean()
    slot_recon_cosine = F.cosine_similarity(slot_recon, slot_keys, dim=-1).mean()

    unique_codes = []
    overlap_scores = []
    jaccard_scores = []
    for q_code, slot_code_row in zip(code_winners, slot_winners):
        slot_set = set(slot_code_row.tolist())
        unique_codes.append(len(slot_set))
        overlap_scores.append(1.0 if int(q_code.item()) in slot_set else 0.0)
        jaccard_scores.append(1.0 / max(len(slot_set), 1) if int(q_code.item()) in slot_set else 0.0)

    values = {
        f"{prefix}codebook_entropy": float(code_entropy.item()),
        f"{prefix}codebook_collision_rate": float(code_collision_rate.item()),
        f"{prefix}dead_code_rate": float(dead_code_rate.item()),
        f"{prefix}codebook_top_margin": float(code_margin.item()),
        f"{prefix}slot_code_entropy": float(slot_entropy.item()),
        f"{prefix}slot_code_collision_rate": float(slot_collision_rate.item()),
        f"{prefix}dead_slot_code_rate": float(dead_slot_code_rate.item()),
        f"{prefix}slot_code_top_margin": float(slot_margin.item()),
        f"{prefix}codebook_recon_mse": float(query_recon_mse.item()),
        f"{prefix}codebook_recon_cosine": float(query_recon_cosine.item()),
        f"{prefix}slot_remap_recon_mse": float(slot_recon_mse.item()),
        f"{prefix}slot_remap_recon_cosine": float(slot_recon_cosine.item()),
        f"{prefix}codebook_support_mean": float(sum(unique_codes) / max(len(unique_codes), 1)),
        f"{prefix}codebook_remap_overlap": float(sum(overlap_scores) / max(len(overlap_scores), 1)),
        f"{prefix}codebook_remap_jaccard": float(sum(jaccard_scores) / max(len(jaccard_scores), 1)),
    }
    return values


def _residual_energy_metrics(
    query_residual: torch.Tensor,
    query_proj: torch.Tensor,
    slot_residual: torch.Tensor,
    slot_keys: torch.Tensor,
    residual_gate: torch.Tensor,
) -> dict[str, float]:
    query_ratio = query_residual.norm(dim=-1) / query_proj.norm(dim=-1).clamp_min(1e-8)
    slot_ratio = slot_residual.norm(dim=-1) / slot_keys.norm(dim=-1).clamp_min(1e-8)
    return {
        "residual_query_energy_ratio": float(query_ratio.mean().item()),
        "residual_slot_energy_ratio": float(slot_ratio.mean().item()),
        "residual_gate": float(residual_gate.item()),
    }


def _calibrate_orthogonal_basis(activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered = activations - activations.mean(dim=0, keepdim=True)
    denom = max(int(centered.shape[0]), 1)
    cov = centered.T @ centered / float(denom)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    basis = eigenvectors[:, order]
    eye = torch.eye(basis.shape[0], device=basis.device, dtype=basis.dtype)
    orthogonality_error = torch.linalg.norm(basis.T @ basis - eye) / float(max(basis.shape[0], 1))
    return basis, eigenvalues, orthogonality_error


def _calibrate_signal_basis(activations: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = labels.view(-1)
    overall_mean = activations.mean(dim=0, keepdim=True)
    signal_cov = torch.zeros(
        activations.shape[-1], activations.shape[-1], device=activations.device, dtype=activations.dtype
    )
    total = max(int(labels.numel()), 1)
    for label in torch.unique(labels, sorted=True):
        mask = labels == label
        if not bool(mask.any()):
            continue
        class_acts = activations[mask]
        class_mean = class_acts.mean(dim=0, keepdim=True)
        delta = (class_mean - overall_mean).squeeze(0)
        signal_cov = signal_cov + float(mask.sum().item()) * torch.outer(delta, delta)
    signal_cov = signal_cov / float(total)
    eigenvalues, eigenvectors = torch.linalg.eigh(signal_cov)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    basis = eigenvectors[:, order]
    eye = torch.eye(basis.shape[0], device=basis.device, dtype=basis.dtype)
    orthogonality_error = torch.linalg.norm(basis.T @ basis - eye) / float(max(basis.shape[0], 1))
    return basis, eigenvalues, orthogonality_error


def _subspace_alignment(left: torch.Tensor, right: torch.Tensor) -> float:
    k = min(left.shape[-1], right.shape[-1], left.shape[0], right.shape[0])
    if k <= 0:
        return 0.0
    overlap = torch.linalg.norm(left[:, :k].T @ right[:, :k], ord="fro").pow(2) / float(k)
    return float(overlap.item())


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


class ConstrainedPreconditionedQueryPoolReadout(nn.Module):
    def __init__(self, dim: int, classes: int, pool_slots: int) -> None:
        super().__init__()
        self.pool_queries = nn.Parameter(torch.randn(pool_slots, dim) / math.sqrt(dim))
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.precondition_raw_scale = nn.Parameter(torch.zeros(dim))
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)
        self.last_atom_weights: torch.Tensor | None = None
        self.last_preconditioned_q: torch.Tensor | None = None
        self.last_original_q: torch.Tensor | None = None
        self.last_precondition_scale: torch.Tensor | None = None

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.query_proj(batch.q)
        scale = 1.0 + 0.25 * torch.tanh(self.precondition_raw_scale)
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
        self.last_precondition_scale = scale.detach()
        return summary, effective_weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


class AsymmetricKVBudgetReadout(nn.Module):
    def __init__(self, dim: int, classes: int, route_budget: int, value_budget: int) -> None:
        super().__init__()
        self.route_budget = int(route_budget)
        self.value_budget = int(value_budget)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)
        self.last_atom_weights: torch.Tensor | None = None
        self.last_route_weights: torch.Tensor | None = None
        self.last_value_weights: torch.Tensor | None = None
        self.last_route_indices: torch.Tensor | None = None
        self.last_value_indices: torch.Tensor | None = None
        self.last_route_summary: torch.Tensor | None = None
        self.last_value_summary: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None

    @staticmethod
    def _topk_weights(logits: torch.Tensor, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
        keep = min(max(int(keep), 1), logits.shape[-1])
        top = torch.topk(logits, k=keep, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(dim=-1, index=top.indices, src=top.values)
        weights = torch.softmax(masked, dim=-1)
        return weights, top.indices

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.query_proj(batch.q)
        scale = math.sqrt(batch.q.shape[-1])
        route_logits = torch.einsum("nd,nsd->ns", q_proj, batch.K) / scale
        value_logits = torch.einsum("nd,nsd->ns", q_proj, batch.V) / scale

        route_weights, route_indices = self._topk_weights(route_logits, self.route_budget)
        value_weights, value_indices = self._topk_weights(value_logits, self.value_budget)

        route_summary = torch.einsum("ns,nsd->nd", route_weights, batch.V)
        value_summary = torch.einsum("ns,nsd->nd", value_weights, batch.V)
        gate = torch.sigmoid(((route_summary - value_summary) * q_proj).sum(dim=-1, keepdim=True) / scale)
        summary = gate * route_summary + (1.0 - gate) * value_summary
        combined_weights = 0.5 * (route_weights + value_weights)

        self.last_atom_weights = combined_weights.detach()
        self.last_route_weights = route_weights.detach()
        self.last_value_weights = value_weights.detach()
        self.last_route_indices = route_indices.detach()
        self.last_value_indices = value_indices.detach()
        self.last_route_summary = route_summary.detach()
        self.last_value_summary = value_summary.detach()
        self.last_gate = gate.detach()
        return summary, combined_weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


class CodebookRemapReadout(nn.Module):
    def __init__(self, dim: int, classes: int, codebook_size: int) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, dim) / math.sqrt(dim))
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)
        self.last_atom_weights: torch.Tensor | None = None
        self.last_code_weights: torch.Tensor | None = None
        self.last_slot_code_weights: torch.Tensor | None = None
        self.last_query_proj: torch.Tensor | None = None
        self.last_slot_keys: torch.Tensor | None = None

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.query_proj(batch.q)
        scale = math.sqrt(batch.q.shape[-1])
        code_logits = torch.einsum("nd,cd->nc", q_proj, self.codebook) / scale
        code_weights = torch.softmax(code_logits, dim=-1)
        slot_code_logits = torch.einsum("nsd,cd->nsc", batch.K, self.codebook) / scale
        slot_code_weights = torch.softmax(slot_code_logits, dim=-1)
        slot_code_summaries = torch.einsum("nsc,nsd->ncd", slot_code_weights, batch.V)
        summary = torch.einsum("nc,ncd->nd", code_weights, slot_code_summaries)
        effective_weights = torch.einsum("nc,nsc->ns", code_weights, slot_code_weights)

        self.last_atom_weights = code_weights.detach()
        self.last_code_weights = code_weights.detach()
        self.last_slot_code_weights = slot_code_weights.detach()
        self.last_query_proj = q_proj.detach()
        self.last_slot_keys = batch.K.detach()
        return summary, effective_weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


class ResidualCodebookRemapReadout(nn.Module):
    def __init__(self, dim: int, classes: int, codebook_size: int, residual_codebook_size: int) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.residual_codebook_size = int(residual_codebook_size)
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, dim) / math.sqrt(dim))
        self.residual_codebook = nn.Parameter(torch.randn(self.residual_codebook_size, dim) / math.sqrt(dim))
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.residual_logit = nn.Parameter(torch.tensor(0.0))
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)
        self.last_atom_weights: torch.Tensor | None = None
        self.last_code_weights: torch.Tensor | None = None
        self.last_slot_code_weights: torch.Tensor | None = None
        self.last_query_proj: torch.Tensor | None = None
        self.last_slot_keys: torch.Tensor | None = None
        self.last_residual_code_weights: torch.Tensor | None = None
        self.last_residual_slot_code_weights: torch.Tensor | None = None
        self.last_query_residual: torch.Tensor | None = None
        self.last_slot_residual: torch.Tensor | None = None
        self.last_residual_gate: torch.Tensor | None = None

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.query_proj(batch.q)
        scale = math.sqrt(batch.q.shape[-1])

        code_logits = torch.einsum("nd,cd->nc", q_proj, self.codebook) / scale
        code_weights = torch.softmax(code_logits, dim=-1)
        slot_code_logits = torch.einsum("nsd,cd->nsc", batch.K, self.codebook) / scale
        slot_code_weights = torch.softmax(slot_code_logits, dim=-1)
        slot_code_summaries = torch.einsum("nsc,nsd->ncd", slot_code_weights, batch.V)
        base_summary = torch.einsum("nc,ncd->nd", code_weights, slot_code_summaries)
        base_weights = torch.einsum("nc,nsc->ns", code_weights, slot_code_weights)

        query_base = code_weights @ self.codebook
        slot_base = torch.einsum("nsc,cd->nsd", slot_code_weights, self.codebook)
        query_residual = q_proj - query_base
        slot_residual = batch.K - slot_base

        residual_code_logits = torch.einsum("nd,rd->nr", query_residual, self.residual_codebook) / scale
        residual_code_weights = torch.softmax(residual_code_logits, dim=-1)
        residual_slot_logits = torch.einsum("nsd,rd->nsr", slot_residual, self.residual_codebook) / scale
        residual_slot_code_weights = torch.softmax(residual_slot_logits, dim=-1)
        residual_slot_summaries = torch.einsum("nsr,nsd->nrd", residual_slot_code_weights, batch.V)
        residual_summary = torch.einsum("nr,nrd->nd", residual_code_weights, residual_slot_summaries)
        residual_weights = torch.einsum("nr,nsr->ns", residual_code_weights, residual_slot_code_weights)

        residual_gate = torch.sigmoid(self.residual_logit)
        summary = base_summary + residual_gate * residual_summary
        effective_weights = (base_weights + residual_gate * residual_weights) / (1.0 + residual_gate)

        self.last_atom_weights = torch.cat([code_weights, residual_code_weights], dim=-1).detach()
        self.last_code_weights = code_weights.detach()
        self.last_slot_code_weights = slot_code_weights.detach()
        self.last_query_proj = q_proj.detach()
        self.last_slot_keys = batch.K.detach()
        self.last_residual_code_weights = residual_code_weights.detach()
        self.last_residual_slot_code_weights = residual_slot_code_weights.detach()
        self.last_query_residual = query_residual.detach()
        self.last_slot_residual = slot_residual.detach()
        self.last_residual_gate = residual_gate.detach()
        return summary, effective_weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


class ProtectedChannelResidualCodebookReadout(nn.Module):
    def __init__(
        self,
        dim: int,
        classes: int,
        codebook_size: int,
        residual_codebook_size: int,
        protected_channels: int,
    ) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.residual_codebook_size = int(residual_codebook_size)
        self.protected_channels = max(1, min(int(protected_channels), dim))
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, dim) / math.sqrt(dim))
        self.protected_residual_codebook = nn.Parameter(
            torch.randn(self.residual_codebook_size, self.protected_channels) / math.sqrt(self.protected_channels)
        )
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.protected_residual_logit = nn.Parameter(torch.tensor(0.0))
        protected_mask = torch.zeros(dim)
        protected_mask[: self.protected_channels] = 1.0
        self.register_buffer("protected_mask", protected_mask)
        self.classifier = nn.Linear(dim, classes)
        self.reconstructor = nn.Linear(dim, dim)
        self.last_atom_weights: torch.Tensor | None = None
        self.last_code_weights: torch.Tensor | None = None
        self.last_slot_code_weights: torch.Tensor | None = None
        self.last_query_proj: torch.Tensor | None = None
        self.last_slot_keys: torch.Tensor | None = None
        self.last_protected_residual_code_weights: torch.Tensor | None = None
        self.last_protected_residual_slot_code_weights: torch.Tensor | None = None
        self.last_query_residual: torch.Tensor | None = None
        self.last_slot_residual: torch.Tensor | None = None
        self.last_protected_query_residual: torch.Tensor | None = None
        self.last_protected_slot_residual: torch.Tensor | None = None
        self.last_protected_gate: torch.Tensor | None = None

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        q_proj = self.query_proj(batch.q)
        scale = math.sqrt(batch.q.shape[-1])

        code_logits = torch.einsum("nd,cd->nc", q_proj, self.codebook) / scale
        code_weights = torch.softmax(code_logits, dim=-1)
        slot_code_logits = torch.einsum("nsd,cd->nsc", batch.K, self.codebook) / scale
        slot_code_weights = torch.softmax(slot_code_logits, dim=-1)
        slot_code_summaries = torch.einsum("nsc,nsd->ncd", slot_code_weights, batch.V)
        base_summary = torch.einsum("nc,ncd->nd", code_weights, slot_code_summaries)
        base_weights = torch.einsum("nc,nsc->ns", code_weights, slot_code_weights)

        query_base = code_weights @ self.codebook
        slot_base = torch.einsum("nsc,cd->nsd", slot_code_weights, self.codebook)
        query_residual = q_proj - query_base
        slot_residual = batch.K - slot_base
        protected_query_residual = (query_residual * self.protected_mask)[:, : self.protected_channels]
        protected_slot_residual = (slot_residual * self.protected_mask)[..., : self.protected_channels]

        protected_residual_code_logits = (
            torch.einsum("nd,rd->nr", protected_query_residual, self.protected_residual_codebook) / scale
        )
        protected_residual_code_weights = torch.softmax(protected_residual_code_logits, dim=-1)
        protected_residual_slot_logits = torch.einsum(
            "nsd,rd->nsr", protected_slot_residual, self.protected_residual_codebook
        ) / scale
        protected_residual_slot_code_weights = torch.softmax(protected_residual_slot_logits, dim=-1)
        protected_slot_summaries = torch.einsum("nsr,nsd->nrd", protected_residual_slot_code_weights, batch.V)
        protected_residual_summary = torch.einsum(
            "nr,nrd->nd", protected_residual_code_weights, protected_slot_summaries
        )
        protected_residual_weights = torch.einsum(
            "nr,nsr->ns", protected_residual_code_weights, protected_residual_slot_code_weights
        )

        protected_gate = torch.sigmoid(self.protected_residual_logit)
        summary = base_summary + protected_gate * protected_residual_summary
        effective_weights = (base_weights + protected_gate * protected_residual_weights) / (1.0 + protected_gate)

        self.last_atom_weights = torch.cat([code_weights, protected_residual_code_weights], dim=-1).detach()
        self.last_code_weights = code_weights.detach()
        self.last_slot_code_weights = slot_code_weights.detach()
        self.last_query_proj = q_proj.detach()
        self.last_slot_keys = batch.K.detach()
        self.last_protected_residual_code_weights = protected_residual_code_weights.detach()
        self.last_protected_residual_slot_code_weights = protected_residual_slot_code_weights.detach()
        self.last_query_residual = query_residual.detach()
        self.last_slot_residual = slot_residual.detach()
        self.last_protected_query_residual = protected_query_residual.detach()
        self.last_protected_slot_residual = protected_slot_residual.detach()
        self.last_protected_gate = protected_gate.detach()
        return summary, effective_weights

    def forward(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary, weights = self.encode(batch)
        return self.classifier(summary), self.reconstructor(summary), weights


class GaugeAwareProtectedChannelResidualCodebookReadout(ProtectedChannelResidualCodebookReadout):
    def __init__(
        self,
        dim: int,
        classes: int,
        codebook_size: int,
        residual_codebook_size: int,
        protected_channels: int,
    ) -> None:
        super().__init__(dim, classes, codebook_size, residual_codebook_size, protected_channels)
        self.register_buffer("gauge_basis", torch.eye(dim))
        self.register_buffer("gauge_eigenvalues", torch.ones(dim))
        self.last_gauge_basis_orthogonality_error: torch.Tensor | None = None
        self.last_gauge_protected_energy_fraction: torch.Tensor | None = None
        self.last_gauge_eigenvalue_top_margin: torch.Tensor | None = None
        self.last_gauge_selected_channels: torch.Tensor | None = None
        self._gauge_calibrated = False

    @torch.no_grad()
    def finalize_calibration(self, batch: ToyBatch) -> None:
        dim = batch.K.shape[-1]
        flat = torch.cat([batch.K.reshape(-1, dim), batch.V.reshape(-1, dim)], dim=0)
        basis, eigenvalues, orthogonality_error = _calibrate_orthogonal_basis(flat)
        self.gauge_basis.copy_(basis)
        self.gauge_eigenvalues.copy_(eigenvalues)
        protected = max(1, min(self.protected_channels, dim))
        protected_energy = (eigenvalues[:protected].sum() / eigenvalues.sum().clamp_min(1e-8)).clamp(0.0, 1.0)
        top_margin = eigenvalues[0] - eigenvalues[min(protected, eigenvalues.numel() - 1)]
        self.last_gauge_basis_orthogonality_error = orthogonality_error.detach()
        self.last_gauge_protected_energy_fraction = protected_energy.detach()
        self.last_gauge_eigenvalue_top_margin = top_margin.detach()
        self.last_gauge_selected_channels = torch.arange(protected, device=batch.K.device)
        self._gauge_calibrated = True

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._gauge_calibrated:
            self.finalize_calibration(batch)

        q_proj = self.query_proj(batch.q) @ self.gauge_basis
        k_cal = torch.einsum("nsd,df->nsf", batch.K, self.gauge_basis)
        v_cal = torch.einsum("nsd,df->nsf", batch.V, self.gauge_basis)
        scale = math.sqrt(batch.q.shape[-1])

        code_logits = torch.einsum("nd,cd->nc", q_proj, self.codebook) / scale
        code_weights = torch.softmax(code_logits, dim=-1)
        slot_code_logits = torch.einsum("nsd,cd->nsc", k_cal, self.codebook) / scale
        slot_code_weights = torch.softmax(slot_code_logits, dim=-1)
        slot_code_summaries = torch.einsum("nsc,nsd->ncd", slot_code_weights, v_cal)
        base_summary = torch.einsum("nc,ncd->nd", code_weights, slot_code_summaries)
        base_weights = torch.einsum("nc,nsc->ns", code_weights, slot_code_weights)

        query_base = code_weights @ self.codebook
        slot_base = torch.einsum("nsc,cd->nsd", slot_code_weights, self.codebook)
        query_residual = q_proj - query_base
        slot_residual = k_cal - slot_base
        protected_query_residual = (query_residual * self.protected_mask)[:, : self.protected_channels]
        protected_slot_residual = (slot_residual * self.protected_mask)[..., : self.protected_channels]

        protected_residual_code_logits = (
            torch.einsum("nd,rd->nr", protected_query_residual, self.protected_residual_codebook) / scale
        )
        protected_residual_code_weights = torch.softmax(protected_residual_code_logits, dim=-1)
        protected_residual_slot_logits = torch.einsum(
            "nsd,rd->nsr", protected_slot_residual, self.protected_residual_codebook
        ) / scale
        protected_residual_slot_code_weights = torch.softmax(protected_residual_slot_logits, dim=-1)
        protected_slot_summaries = torch.einsum("nsr,nsd->nrd", protected_residual_slot_code_weights, v_cal)
        protected_residual_summary = torch.einsum(
            "nr,nrd->nd", protected_residual_code_weights, protected_slot_summaries
        )
        protected_residual_weights = torch.einsum(
            "nr,nsr->ns", protected_residual_code_weights, protected_residual_slot_code_weights
        )

        protected_gate = torch.sigmoid(self.protected_residual_logit)
        summary = base_summary + protected_gate * protected_residual_summary
        effective_weights = (base_weights + protected_gate * protected_residual_weights) / (1.0 + protected_gate)

        self.last_atom_weights = torch.cat([code_weights, protected_residual_code_weights], dim=-1).detach()
        self.last_code_weights = code_weights.detach()
        self.last_slot_code_weights = slot_code_weights.detach()
        self.last_query_proj = q_proj.detach()
        self.last_slot_keys = k_cal.detach()
        self.last_protected_residual_code_weights = protected_residual_code_weights.detach()
        self.last_protected_residual_slot_code_weights = protected_residual_slot_code_weights.detach()
        self.last_query_residual = query_residual.detach()
        self.last_slot_residual = slot_residual.detach()
        self.last_protected_query_residual = protected_query_residual.detach()
        self.last_protected_slot_residual = protected_slot_residual.detach()
        self.last_protected_gate = protected_gate.detach()
        return summary, effective_weights


class SignalAwareProtectedChannelResidualCodebookReadout(ProtectedChannelResidualCodebookReadout):
    def __init__(
        self,
        dim: int,
        classes: int,
        codebook_size: int,
        residual_codebook_size: int,
        protected_channels: int,
    ) -> None:
        super().__init__(dim, classes, codebook_size, residual_codebook_size, protected_channels)
        self.register_buffer("signal_basis", torch.eye(dim))
        self.register_buffer("signal_eigenvalues", torch.ones(dim))
        self.register_buffer("variance_basis", torch.eye(dim))
        self.register_buffer("variance_eigenvalues", torch.ones(dim))
        self.last_gauge_basis_orthogonality_error: torch.Tensor | None = None
        self.last_gauge_protected_energy_fraction: torch.Tensor | None = None
        self.last_gauge_eigenvalue_top_margin: torch.Tensor | None = None
        self.last_gauge_selected_channels: torch.Tensor | None = None
        self.last_signal_basis_orthogonality_error: torch.Tensor | None = None
        self.last_signal_task_energy_fraction: torch.Tensor | None = None
        self.last_signal_eigenvalue_top_margin: torch.Tensor | None = None
        self.last_signal_selected_channels: torch.Tensor | None = None
        self.last_signal_variance_alignment: torch.Tensor | None = None
        self._signal_calibrated = False

    @torch.no_grad()
    def finalize_calibration(self, batch: ToyBatch) -> None:
        dim = batch.q.shape[-1]
        signal_basis, signal_eigenvalues, signal_orthogonality_error = _calibrate_signal_basis(batch.q, batch.y)
        variance_basis, variance_eigenvalues, variance_orthogonality_error = _calibrate_orthogonal_basis(batch.q)
        self.signal_basis.copy_(signal_basis)
        self.signal_eigenvalues.copy_(signal_eigenvalues)
        self.variance_basis.copy_(variance_basis)
        self.variance_eigenvalues.copy_(variance_eigenvalues)
        protected = max(1, min(self.protected_channels, dim))
        variance_energy = (
            variance_eigenvalues[:protected].sum() / variance_eigenvalues.sum().clamp_min(1e-8)
        ).clamp(0.0, 1.0)
        variance_top_margin = variance_eigenvalues[0] - variance_eigenvalues[min(protected, variance_eigenvalues.numel() - 1)]
        signal_energy = (signal_eigenvalues[:protected].sum() / signal_eigenvalues.sum().clamp_min(1e-8)).clamp(0.0, 1.0)
        top_margin = signal_eigenvalues[0] - signal_eigenvalues[min(protected, signal_eigenvalues.numel() - 1)]
        alignment = _subspace_alignment(signal_basis[:, :protected], variance_basis[:, :protected])
        self.last_gauge_basis_orthogonality_error = variance_orthogonality_error.detach()
        self.last_gauge_protected_energy_fraction = variance_energy.detach()
        self.last_gauge_eigenvalue_top_margin = variance_top_margin.detach()
        self.last_gauge_selected_channels = torch.arange(protected, device=batch.q.device)
        self.last_signal_basis_orthogonality_error = signal_orthogonality_error.detach()
        self.last_signal_task_energy_fraction = signal_energy.detach()
        self.last_signal_eigenvalue_top_margin = top_margin.detach()
        self.last_signal_selected_channels = torch.arange(protected, device=batch.q.device)
        self.last_signal_variance_alignment = torch.tensor(alignment, device=batch.q.device, dtype=batch.q.dtype)
        self._signal_calibrated = True

    def encode(self, batch: ToyBatch) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._signal_calibrated:
            self.finalize_calibration(batch)

        q_proj = self.query_proj(batch.q) @ self.signal_basis
        k_cal = torch.einsum("nsd,df->nsf", batch.K, self.signal_basis)
        v_cal = torch.einsum("nsd,df->nsf", batch.V, self.signal_basis)
        scale = math.sqrt(batch.q.shape[-1])

        code_logits = torch.einsum("nd,cd->nc", q_proj, self.codebook) / scale
        code_weights = torch.softmax(code_logits, dim=-1)
        slot_code_logits = torch.einsum("nsd,cd->nsc", k_cal, self.codebook) / scale
        slot_code_weights = torch.softmax(slot_code_logits, dim=-1)
        slot_code_summaries = torch.einsum("nsc,nsd->ncd", slot_code_weights, v_cal)
        base_summary = torch.einsum("nc,ncd->nd", code_weights, slot_code_summaries)
        base_weights = torch.einsum("nc,nsc->ns", code_weights, slot_code_weights)

        query_base = code_weights @ self.codebook
        slot_base = torch.einsum("nsc,cd->nsd", slot_code_weights, self.codebook)
        query_residual = q_proj - query_base
        slot_residual = k_cal - slot_base
        protected_query_residual = (query_residual * self.protected_mask)[:, : self.protected_channels]
        protected_slot_residual = (slot_residual * self.protected_mask)[..., : self.protected_channels]

        protected_residual_code_logits = (
            torch.einsum("nd,rd->nr", protected_query_residual, self.protected_residual_codebook) / scale
        )
        protected_residual_code_weights = torch.softmax(protected_residual_code_logits, dim=-1)
        protected_residual_slot_logits = torch.einsum(
            "nsd,rd->nsr", protected_slot_residual, self.protected_residual_codebook
        ) / scale
        protected_residual_slot_code_weights = torch.softmax(protected_residual_slot_logits, dim=-1)
        protected_slot_summaries = torch.einsum("nsr,nsd->nrd", protected_residual_slot_code_weights, v_cal)
        protected_residual_summary = torch.einsum(
            "nr,nrd->nd", protected_residual_code_weights, protected_slot_summaries
        )
        protected_residual_weights = torch.einsum(
            "nr,nsr->ns", protected_residual_code_weights, protected_residual_slot_code_weights
        )

        protected_gate = torch.sigmoid(self.protected_residual_logit)
        summary = base_summary + protected_gate * protected_residual_summary
        effective_weights = (base_weights + protected_gate * protected_residual_weights) / (1.0 + protected_gate)

        self.last_atom_weights = torch.cat([code_weights, protected_residual_code_weights], dim=-1).detach()
        self.last_code_weights = code_weights.detach()
        self.last_slot_code_weights = slot_code_weights.detach()
        self.last_query_proj = q_proj.detach()
        self.last_slot_keys = k_cal.detach()
        self.last_protected_residual_code_weights = protected_residual_code_weights.detach()
        self.last_protected_residual_slot_code_weights = protected_residual_slot_code_weights.detach()
        self.last_query_residual = query_residual.detach()
        self.last_slot_residual = slot_residual.detach()
        self.last_protected_query_residual = protected_query_residual.detach()
        self.last_protected_slot_residual = protected_slot_residual.detach()
        self.last_protected_gate = protected_gate.detach()
        return summary, effective_weights


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
    constrained_scale = getattr(model, "last_precondition_scale", None)
    if preconditioned_q is None or original_q is None:
        metrics.update(
            {
                "precondition_condition_proxy": None,
                "precondition_cosine_drift": None,
                "precondition_norm_ratio": None,
                "precondition_abs_scale_ratio": None,
                "constrained_scale_min": None,
                "constrained_scale_max": None,
                "constrained_scale_mean": None,
            }
        )
    else:
        metrics.update(_precondition_metrics(preconditioned_q, original_q))
        if constrained_scale is None:
            metrics.update(
                {
                    "constrained_scale_min": None,
                    "constrained_scale_max": None,
                    "constrained_scale_mean": None,
                }
            )
        else:
            metrics.update(
                {
                    "constrained_scale_min": float(constrained_scale.min().item()),
                    "constrained_scale_max": float(constrained_scale.max().item()),
                    "constrained_scale_mean": float(constrained_scale.mean().item()),
                }
            )
    route_weights = getattr(model, "last_route_weights", None)
    value_weights = getattr(model, "last_value_weights", None)
    route_indices = getattr(model, "last_route_indices", None)
    value_indices = getattr(model, "last_value_indices", None)
    route_summary = getattr(model, "last_route_summary", None)
    value_summary = getattr(model, "last_value_summary", None)
    gate = getattr(model, "last_gate", None)
    if (
        route_weights is None
        or value_weights is None
        or route_indices is None
        or value_indices is None
        or route_summary is None
        or value_summary is None
        or gate is None
    ):
        metrics.update(
            {
                "kv_route_entropy": None,
                "kv_value_entropy": None,
                "kv_route_collision_rate": None,
                "kv_value_collision_rate": None,
                "kv_route_dead_rate": None,
                "kv_value_dead_rate": None,
                "kv_route_top_margin": None,
                "kv_value_top_margin": None,
                "kv_route_keep_fraction": None,
                "kv_value_keep_fraction": None,
                "kv_route_value_overlap": None,
                "kv_route_value_jaccard": None,
                "kv_route_value_kl": None,
                "kv_route_value_cosine": None,
                "kv_route_value_gap": None,
                "kv_gate_mean": None,
                "kv_gate_std": None,
            }
        )
    else:
        metrics.update(
            _kv_asymmetry_metrics(
                route_weights,
                value_weights,
                route_indices,
                value_indices,
                route_summary,
                value_summary,
                gate,
            )
        )
    code_weights = getattr(model, "last_code_weights", None)
    slot_code_weights = getattr(model, "last_slot_code_weights", None)
    query_proj = getattr(model, "last_query_proj", None)
    slot_keys = getattr(model, "last_slot_keys", None)
    codebook = getattr(model, "codebook", None)
    if code_weights is None or slot_code_weights is None or query_proj is None or slot_keys is None or codebook is None:
        metrics.update(
            {
                "codebook_entropy": None,
                "codebook_collision_rate": None,
                "dead_code_rate": None,
                "codebook_top_margin": None,
                "slot_code_entropy": None,
                "slot_code_collision_rate": None,
                "dead_slot_code_rate": None,
                "slot_code_top_margin": None,
                "codebook_recon_mse": None,
                "codebook_recon_cosine": None,
                "slot_remap_recon_mse": None,
                "slot_remap_recon_cosine": None,
                "codebook_support_mean": None,
                "codebook_remap_overlap": None,
                "codebook_remap_jaccard": None,
            }
        )
    else:
        metrics.update(
            _codebook_metrics(
                code_weights,
                slot_code_weights,
                codebook.detach(),
                query_proj,
                slot_keys,
            )
        )
    residual_code_weights = getattr(model, "last_residual_code_weights", None)
    residual_slot_code_weights = getattr(model, "last_residual_slot_code_weights", None)
    residual_codebook = getattr(model, "residual_codebook", None)
    query_residual = getattr(model, "last_query_residual", None)
    slot_residual = getattr(model, "last_slot_residual", None)
    residual_gate = getattr(model, "last_residual_gate", None)
    if (
        residual_code_weights is None
        or residual_slot_code_weights is None
        or residual_codebook is None
        or query_residual is None
        or slot_residual is None
        or query_proj is None
        or slot_keys is None
        or residual_gate is None
    ):
        metrics.update(
            {
                "residual_codebook_entropy": None,
                "residual_codebook_collision_rate": None,
                "residual_dead_code_rate": None,
                "residual_codebook_top_margin": None,
                "residual_slot_code_entropy": None,
                "residual_slot_code_collision_rate": None,
                "residual_dead_slot_code_rate": None,
                "residual_slot_code_top_margin": None,
                "residual_codebook_recon_mse": None,
                "residual_codebook_recon_cosine": None,
                "residual_slot_remap_recon_mse": None,
                "residual_slot_remap_recon_cosine": None,
                "residual_codebook_support_mean": None,
                "residual_codebook_remap_overlap": None,
                "residual_codebook_remap_jaccard": None,
                "residual_query_energy_ratio": None,
                "residual_slot_energy_ratio": None,
                "residual_gate": None,
            }
        )
    else:
        metrics.update(
            _codebook_metrics(
                residual_code_weights,
                residual_slot_code_weights,
                residual_codebook.detach(),
                query_residual,
                slot_residual,
                prefix="residual_",
            )
        )
        metrics.update(_residual_energy_metrics(query_residual, query_proj, slot_residual, slot_keys, residual_gate))
    signal_basis_orthogonality_error = getattr(model, "last_signal_basis_orthogonality_error", None)
    signal_task_energy_fraction = getattr(model, "last_signal_task_energy_fraction", None)
    signal_eigenvalue_top_margin = getattr(model, "last_signal_eigenvalue_top_margin", None)
    signal_selected_channels = getattr(model, "last_signal_selected_channels", None)
    signal_variance_alignment = getattr(model, "last_signal_variance_alignment", None)
    signal_residual_code_weights = getattr(model, "last_protected_residual_code_weights", None)
    signal_residual_slot_code_weights = getattr(model, "last_protected_residual_slot_code_weights", None)
    signal_residual_codebook = getattr(model, "protected_residual_codebook", None)
    signal_query_residual = getattr(model, "last_protected_query_residual", None)
    signal_slot_residual = getattr(model, "last_protected_slot_residual", None)
    signal_query_full = getattr(model, "last_query_residual", None)
    signal_slot_full = getattr(model, "last_slot_residual", None)
    signal_gate = getattr(model, "last_protected_gate", None)
    signal_channels = getattr(model, "protected_channels", None)
    if (
        signal_basis_orthogonality_error is not None
        and signal_task_energy_fraction is not None
        and signal_eigenvalue_top_margin is not None
        and signal_selected_channels is not None
        and signal_variance_alignment is not None
        and signal_residual_code_weights is not None
        and signal_residual_slot_code_weights is not None
        and signal_residual_codebook is not None
        and signal_query_residual is not None
        and signal_slot_residual is not None
        and signal_query_full is not None
        and signal_slot_full is not None
        and signal_gate is not None
        and signal_channels is not None
    ):
        metrics.update(
            _codebook_metrics(
                signal_residual_code_weights,
                signal_residual_slot_code_weights,
                signal_residual_codebook.detach(),
                signal_query_residual,
                signal_slot_residual,
                prefix="signal_",
            )
        )
        signal_query_ratio = signal_query_residual.norm(dim=-1) / signal_query_full.norm(dim=-1).clamp_min(1e-8)
        signal_slot_ratio = signal_slot_residual.norm(dim=-1) / signal_slot_full.norm(dim=-1).clamp_min(1e-8)
        metrics.update(
            {
                "protected_residual_codebook_entropy": float(metrics["signal_codebook_entropy"]),
                "protected_residual_codebook_collision_rate": float(metrics["signal_codebook_collision_rate"]),
                "protected_dead_code_rate": float(metrics["signal_dead_code_rate"]),
                "protected_residual_codebook_top_margin": float(metrics["signal_codebook_top_margin"]),
                "protected_residual_slot_code_entropy": float(metrics["signal_slot_code_entropy"]),
                "protected_residual_slot_code_collision_rate": float(metrics["signal_slot_code_collision_rate"]),
                "protected_dead_slot_code_rate": float(metrics["signal_dead_slot_code_rate"]),
                "protected_residual_slot_code_top_margin": float(metrics["signal_slot_code_top_margin"]),
                "protected_residual_codebook_recon_mse": float(metrics["signal_codebook_recon_mse"]),
                "protected_residual_codebook_recon_cosine": float(metrics["signal_codebook_recon_cosine"]),
                "protected_residual_slot_remap_recon_mse": float(metrics["signal_slot_remap_recon_mse"]),
                "protected_residual_slot_remap_recon_cosine": float(metrics["signal_slot_remap_recon_cosine"]),
                "protected_residual_codebook_support_mean": float(metrics["signal_codebook_support_mean"]),
                "protected_residual_codebook_remap_overlap": float(metrics["signal_codebook_remap_overlap"]),
                "protected_residual_codebook_remap_jaccard": float(metrics["signal_codebook_remap_jaccard"]),
                "protected_query_energy_ratio": float(signal_query_ratio.mean().item()),
                "protected_slot_energy_ratio": float(signal_slot_ratio.mean().item()),
                "protected_gate": float(signal_gate.item()),
                "signal_query_energy_ratio": float(signal_query_ratio.mean().item()),
                "signal_slot_energy_ratio": float(signal_slot_ratio.mean().item()),
                "signal_gate": float(signal_gate.item()),
                "signal_basis_orthogonality_error": float(signal_basis_orthogonality_error.item()),
                "signal_task_energy_fraction": float(signal_task_energy_fraction.item()),
                "signal_eigenvalue_top_margin": float(signal_eigenvalue_top_margin.item()),
                "signal_selected_channels": float(signal_selected_channels.numel())
                if isinstance(signal_selected_channels, torch.Tensor)
                else None,
                "signal_variance_alignment": float(signal_variance_alignment.item()),
                "gauge_basis_orthogonality_error": float(
                    getattr(model, "last_gauge_basis_orthogonality_error").item()
                )
                if getattr(model, "last_gauge_basis_orthogonality_error", None) is not None
                else None,
                "gauge_protected_energy_fraction": float(
                    getattr(model, "last_gauge_protected_energy_fraction").item()
                )
                if getattr(model, "last_gauge_protected_energy_fraction", None) is not None
                else None,
                "gauge_eigenvalue_top_margin": float(getattr(model, "last_gauge_eigenvalue_top_margin").item())
                if getattr(model, "last_gauge_eigenvalue_top_margin", None) is not None
                else None,
                "gauge_selected_channels": float(getattr(model, "last_gauge_selected_channels").numel())
                if isinstance(getattr(model, "last_gauge_selected_channels", None), torch.Tensor)
                else None,
            }
        )
    else:
        protected_residual_code_weights = getattr(model, "last_protected_residual_code_weights", None)
        protected_residual_slot_code_weights = getattr(model, "last_protected_residual_slot_code_weights", None)
        protected_residual_codebook = getattr(model, "protected_residual_codebook", None)
        protected_query_residual = getattr(model, "last_protected_query_residual", None)
        protected_slot_residual = getattr(model, "last_protected_slot_residual", None)
        protected_query_full = getattr(model, "last_query_residual", None)
        protected_slot_full = getattr(model, "last_slot_residual", None)
        protected_gate = getattr(model, "last_protected_gate", None)
        protected_channels = getattr(model, "protected_channels", None)
        gauge_basis_orthogonality_error = getattr(model, "last_gauge_basis_orthogonality_error", None)
        gauge_protected_energy_fraction = getattr(model, "last_gauge_protected_energy_fraction", None)
        gauge_eigenvalue_top_margin = getattr(model, "last_gauge_eigenvalue_top_margin", None)
        gauge_selected_channels = getattr(model, "last_gauge_selected_channels", None)
        if (
            protected_residual_code_weights is None
            or protected_residual_slot_code_weights is None
            or protected_residual_codebook is None
            or protected_query_residual is None
            or protected_slot_residual is None
            or protected_query_full is None
            or protected_slot_full is None
            or protected_gate is None
            or protected_channels is None
        ):
            metrics.update(
                {
                    "protected_residual_codebook_entropy": None,
                    "protected_residual_codebook_collision_rate": None,
                    "protected_dead_code_rate": None,
                    "protected_residual_codebook_top_margin": None,
                    "protected_residual_slot_code_entropy": None,
                    "protected_residual_slot_code_collision_rate": None,
                    "protected_dead_slot_code_rate": None,
                    "protected_residual_slot_code_top_margin": None,
                    "protected_residual_codebook_recon_mse": None,
                    "protected_residual_codebook_recon_cosine": None,
                    "protected_residual_slot_remap_recon_mse": None,
                    "protected_residual_slot_remap_recon_cosine": None,
                    "protected_residual_codebook_support_mean": None,
                    "protected_residual_codebook_remap_overlap": None,
                    "protected_residual_codebook_remap_jaccard": None,
                    "protected_query_energy_ratio": None,
                    "protected_slot_energy_ratio": None,
                    "protected_gate": None,
                    "gauge_basis_orthogonality_error": None,
                    "gauge_protected_energy_fraction": None,
                    "gauge_eigenvalue_top_margin": None,
                    "gauge_selected_channels": None,
                }
            )
        else:
            metrics.update(
                _codebook_metrics(
                    protected_residual_code_weights,
                    protected_residual_slot_code_weights,
                    protected_residual_codebook.detach(),
                    protected_query_residual,
                    protected_slot_residual,
                    prefix="protected_residual_",
                )
            )
            protected_query_ratio = protected_query_residual.norm(dim=-1) / protected_query_full.norm(dim=-1).clamp_min(1e-8)
            protected_slot_ratio = protected_slot_residual.norm(dim=-1) / protected_slot_full.norm(dim=-1).clamp_min(1e-8)
            metrics.update(
                {
                    "protected_query_energy_ratio": float(protected_query_ratio.mean().item()),
                    "protected_slot_energy_ratio": float(protected_slot_ratio.mean().item()),
                    "protected_gate": float(protected_gate.item()),
                }
            )
            if gauge_basis_orthogonality_error is None or gauge_protected_energy_fraction is None or gauge_eigenvalue_top_margin is None:
                metrics.update(
                    {
                        "gauge_basis_orthogonality_error": None,
                        "gauge_protected_energy_fraction": None,
                        "gauge_eigenvalue_top_margin": None,
                        "gauge_selected_channels": None,
                    }
                )
            else:
                metrics.update(
                    {
                        "gauge_basis_orthogonality_error": float(gauge_basis_orthogonality_error.item()),
                        "gauge_protected_energy_fraction": float(gauge_protected_energy_fraction.item()),
                        "gauge_eigenvalue_top_margin": float(gauge_eigenvalue_top_margin.item()),
                        "gauge_selected_channels": float(gauge_selected_channels.numel())
                        if isinstance(gauge_selected_channels, torch.Tensor)
                        else None,
                    }
                )
    return metrics


def run_experiment(config: ToyConfig, scenarios: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = _make_base_tensors(config)
    protected_channels = max(1, min(config.protected_channels, config.dim))
    method_seed_offsets = {
        "topk": 101,
        "query_pool": 211,
        "preconditioned_query_pool": 307,
        "constrained_preconditioned_query_pool": 359,
        "route_atom": 401,
        "asymmetric_kv_budget": 463,
        "codebook_remap": 509,
        "residual_codebook_remap": 557,
        "protected_channel_residual_codebook_remap": 619,
        "gauge_aware_protected_channel_residual_codebook_remap": 683,
        "signal_aware_protected_channel_residual_codebook_remap": 739,
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
            (
                "constrained_preconditioned_query_pool",
                lambda: ConstrainedPreconditionedQueryPoolReadout(config.dim, config.classes, config.pool_slots),
                1,
            ),
            ("route_atom", lambda: RouteAtomReadout(config.dim, config.classes, config.route_atoms), 1),
            (
                "asymmetric_kv_budget",
                lambda: AsymmetricKVBudgetReadout(
                    config.dim, config.classes, config.kv_route_budget, config.kv_value_budget
                ),
                config.kv_route_budget + config.kv_value_budget,
            ),
            (
                "codebook_remap",
                lambda: CodebookRemapReadout(config.dim, config.classes, config.codebook_size),
                config.codebook_size,
            ),
            (
                "residual_codebook_remap",
                lambda: ResidualCodebookRemapReadout(
                    config.dim, config.classes, config.codebook_size, config.residual_codebook_size
                ),
                config.codebook_size + config.residual_codebook_size,
            ),
            (
                "protected_channel_residual_codebook_remap",
                lambda: ProtectedChannelResidualCodebookReadout(
                    config.dim,
                    config.classes,
                    config.codebook_size,
                    config.residual_codebook_size,
                    protected_channels,
                ),
                config.codebook_size + config.residual_codebook_size,
            ),
            (
                "gauge_aware_protected_channel_residual_codebook_remap",
                lambda: GaugeAwareProtectedChannelResidualCodebookReadout(
                    config.dim,
                    config.classes,
                    config.codebook_size,
                    config.residual_codebook_size,
                    protected_channels,
                ),
                config.codebook_size + config.residual_codebook_size,
            ),
            (
                "signal_aware_protected_channel_residual_codebook_remap",
                lambda: SignalAwareProtectedChannelResidualCodebookReadout(
                    config.dim,
                    config.classes,
                    config.codebook_size,
                    config.residual_codebook_size,
                    protected_channels,
                ),
                config.codebook_size + config.residual_codebook_size,
            ),
        ]
        for method, factory, margin_k in methods:
            torch.manual_seed(config.seed + scenario_idx * 10_000 + method_seed_offsets[method])
            model = factory()
            if hasattr(model, "finalize_calibration"):
                model.finalize_calibration(train)
            _train_model(model, train, config)
            metrics = _evaluate_model(model, test, margin_k=margin_k)
            if method == "topk":
                budget = config.top_k
            elif method in {"query_pool", "preconditioned_query_pool", "constrained_preconditioned_query_pool"}:
                budget = config.pool_slots
            elif method == "route_atom":
                budget = config.route_atoms
            elif method == "codebook_remap":
                budget = config.codebook_size
            elif method == "residual_codebook_remap":
                budget = config.codebook_size + config.residual_codebook_size
            elif method == "protected_channel_residual_codebook_remap":
                budget = config.codebook_size + config.residual_codebook_size
            elif method == "gauge_aware_protected_channel_residual_codebook_remap":
                budget = config.codebook_size + config.residual_codebook_size
            elif method == "signal_aware_protected_channel_residual_codebook_remap":
                budget = config.codebook_size + config.residual_codebook_size
            else:
                budget = config.kv_route_budget + config.kv_value_budget
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "budget": budget,
                    "kv_route_budget": config.kv_route_budget if method == "asymmetric_kv_budget" else None,
                    "kv_value_budget": config.kv_value_budget if method == "asymmetric_kv_budget" else None,
                    "codebook_size": config.codebook_size
                    if method
                    in {
                        "codebook_remap",
                        "residual_codebook_remap",
                        "protected_channel_residual_codebook_remap",
                        "gauge_aware_protected_channel_residual_codebook_remap",
                        "signal_aware_protected_channel_residual_codebook_remap",
                    }
                    else None,
                    "residual_codebook_size": config.residual_codebook_size
                    if method
                    in {
                        "residual_codebook_remap",
                        "protected_channel_residual_codebook_remap",
                        "gauge_aware_protected_channel_residual_codebook_remap",
                        "signal_aware_protected_channel_residual_codebook_remap",
                    }
                    else None,
                    "protected_channels": protected_channels
                    if method
                    in {
                        "protected_channel_residual_codebook_remap",
                        "gauge_aware_protected_channel_residual_codebook_remap",
                        "signal_aware_protected_channel_residual_codebook_remap",
                    }
                    else None,
                    "protected_channel_fraction": protected_channels / config.dim
                    if method in {
                        "protected_channel_residual_codebook_remap",
                        "gauge_aware_protected_channel_residual_codebook_remap",
                        "signal_aware_protected_channel_residual_codebook_remap",
                    }
                    else None,
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
        "| Scenario | Method | Budget | KV route budget | KV value budget | Codebook size | Residual codebook size | Task acc | Rec MSE | Route entropy | Collision | Dead slots | Top margin | Atom entropy | Atom collision | Dead atoms | Atom margin | Precond cond. | Cosine drift | Norm ratio | Abs scale ratio | KV route entropy | KV value entropy | KV route collision | KV value collision | KV route dead | KV value dead | KV route margin | KV value margin | KV overlap | KV Jaccard | KV KL | KV cosine | KV gap | KV gate mean | KV gate std | Constrained scale min | Constrained scale max | Constrained scale mean | Codebook entropy | Codebook collision | Dead codes | Codebook margin | Slot code entropy | Slot code collision | Dead slot codes | Slot code margin | Codebook recon MSE | Codebook recon cosine | Slot remap recon MSE | Slot remap recon cosine | Codebook support | Remap overlap | Remap Jaccard | Residual codebook entropy | Residual codebook collision | Residual dead codes | Residual codebook margin | Residual slot code entropy | Residual slot code collision | Residual dead slot codes | Residual slot code margin | Residual recon MSE | Residual recon cosine | Residual slot recon MSE | Residual slot recon cosine | Residual support | Residual overlap | Residual Jaccard | Residual query energy | Residual slot energy | Residual gate | Protected channels | Protected fraction | Protected residual entropy | Protected residual collision | Protected query energy | Protected slot energy | Protected gate | Gauge orth. err. | Gauge energy frac. | Gauge top margin | Gauge selected | Signal orth. err. | Signal task energy | Signal top margin | Signal selected | Signal var. align | Signal query energy | Signal slot energy | Signal gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            (
                "| {scenario} | {method} | {budget} | {kv_route_budget} | {kv_value_budget} | {codebook_size} | {residual_codebook_size} | {task_acc} | {rec_mse} | "
                "{route_entropy} | {slot_collision_rate} | {dead_slot_rate} | {top_margin} | "
                "{atom_entropy} | {atom_collision_rate} | {dead_atom_rate} | {atom_top_margin} | "
                "{precondition_condition_proxy} | {precondition_cosine_drift} | {precondition_norm_ratio} | {precondition_abs_scale_ratio} | "
                "{kv_route_entropy} | {kv_value_entropy} | {kv_route_collision_rate} | {kv_value_collision_rate} | "
                "{kv_route_dead_rate} | {kv_value_dead_rate} | {kv_route_top_margin} | {kv_value_top_margin} | "
                "{kv_route_value_overlap} | {kv_route_value_jaccard} | {kv_route_value_kl} | {kv_route_value_cosine} | "
                "{kv_route_value_gap} | {kv_gate_mean} | {kv_gate_std} | {constrained_scale_min} | {constrained_scale_max} | {constrained_scale_mean} | "
                "{codebook_entropy} | {codebook_collision_rate} | {dead_code_rate} | {codebook_top_margin} | "
                "{slot_code_entropy} | {slot_code_collision_rate} | {dead_slot_code_rate} | {slot_code_top_margin} | "
                "{codebook_recon_mse} | {codebook_recon_cosine} | {slot_remap_recon_mse} | {slot_remap_recon_cosine} | "
                "{codebook_support_mean} | {codebook_remap_overlap} | {codebook_remap_jaccard} | "
                "{residual_codebook_entropy} | {residual_codebook_collision_rate} | {residual_dead_code_rate} | "
                "{residual_codebook_top_margin} | {residual_slot_code_entropy} | {residual_slot_code_collision_rate} | "
                "{residual_dead_slot_code_rate} | {residual_slot_code_top_margin} | {residual_codebook_recon_mse} | "
                "{residual_codebook_recon_cosine} | {residual_slot_remap_recon_mse} | {residual_slot_remap_recon_cosine} | "
                "{residual_codebook_support_mean} | {residual_codebook_remap_overlap} | {residual_codebook_remap_jaccard} | "
                "{residual_query_energy_ratio} | {residual_slot_energy_ratio} | {residual_gate} | "
                "{protected_channels} | {protected_channel_fraction} | {protected_residual_codebook_entropy} | "
                "{protected_residual_codebook_collision_rate} | {protected_query_energy_ratio} | "
                "{protected_slot_energy_ratio} | {protected_gate} | {gauge_basis_orthogonality_error} | "
                "{gauge_protected_energy_fraction} | {gauge_eigenvalue_top_margin} | {gauge_selected_channels} | "
                "{signal_basis_orthogonality_error} | {signal_task_energy_fraction} | {signal_eigenvalue_top_margin} | "
                "{signal_selected_channels} | {signal_variance_alignment} | {signal_query_energy_ratio} | "
                "{signal_slot_energy_ratio} | {signal_gate} |"
            ).format(
                scenario=row["scenario"],
                method=row["method"],
                budget=row["budget"],
                kv_route_budget=fmt(row.get("kv_route_budget")),
                kv_value_budget=fmt(row.get("kv_value_budget")),
                codebook_size=fmt(row.get("codebook_size")),
                residual_codebook_size=fmt(row.get("residual_codebook_size")),
                task_acc=fmt(row.get("task_acc")),
                rec_mse=fmt(row.get("rec_mse")),
                route_entropy=fmt(row.get("route_entropy")),
                slot_collision_rate=fmt(row.get("slot_collision_rate")),
                dead_slot_rate=fmt(row.get("dead_slot_rate")),
                top_margin=fmt(row.get("top_margin")),
                atom_entropy=fmt(row.get("atom_entropy")),
                atom_collision_rate=fmt(row.get("atom_collision_rate")),
                dead_atom_rate=fmt(row.get("dead_atom_rate")),
                atom_top_margin=fmt(row.get("atom_top_margin")),
                precondition_condition_proxy=fmt(row.get("precondition_condition_proxy")),
                precondition_cosine_drift=fmt(row.get("precondition_cosine_drift")),
                precondition_norm_ratio=fmt(row.get("precondition_norm_ratio")),
                precondition_abs_scale_ratio=fmt(row.get("precondition_abs_scale_ratio")),
                kv_route_entropy=fmt(row.get("kv_route_entropy")),
                kv_value_entropy=fmt(row.get("kv_value_entropy")),
                kv_route_collision_rate=fmt(row.get("kv_route_collision_rate")),
                kv_value_collision_rate=fmt(row.get("kv_value_collision_rate")),
                kv_route_dead_rate=fmt(row.get("kv_route_dead_rate")),
                kv_value_dead_rate=fmt(row.get("kv_value_dead_rate")),
                kv_route_top_margin=fmt(row.get("kv_route_top_margin")),
                kv_value_top_margin=fmt(row.get("kv_value_top_margin")),
                kv_route_value_overlap=fmt(row.get("kv_route_value_overlap")),
                kv_route_value_jaccard=fmt(row.get("kv_route_value_jaccard")),
                kv_route_value_kl=fmt(row.get("kv_route_value_kl")),
                kv_route_value_cosine=fmt(row.get("kv_route_value_cosine")),
                kv_route_value_gap=fmt(row.get("kv_route_value_gap")),
                kv_gate_mean=fmt(row.get("kv_gate_mean")),
                kv_gate_std=fmt(row.get("kv_gate_std")),
                constrained_scale_min=fmt(row.get("constrained_scale_min")),
                constrained_scale_max=fmt(row.get("constrained_scale_max")),
                constrained_scale_mean=fmt(row.get("constrained_scale_mean")),
                codebook_entropy=fmt(row.get("codebook_entropy")),
                codebook_collision_rate=fmt(row.get("codebook_collision_rate")),
                dead_code_rate=fmt(row.get("dead_code_rate")),
                codebook_top_margin=fmt(row.get("codebook_top_margin")),
                slot_code_entropy=fmt(row.get("slot_code_entropy")),
                slot_code_collision_rate=fmt(row.get("slot_code_collision_rate")),
                dead_slot_code_rate=fmt(row.get("dead_slot_code_rate")),
                slot_code_top_margin=fmt(row.get("slot_code_top_margin")),
                codebook_recon_mse=fmt(row.get("codebook_recon_mse")),
                codebook_recon_cosine=fmt(row.get("codebook_recon_cosine")),
                slot_remap_recon_mse=fmt(row.get("slot_remap_recon_mse")),
                slot_remap_recon_cosine=fmt(row.get("slot_remap_recon_cosine")),
                codebook_support_mean=fmt(row.get("codebook_support_mean")),
                codebook_remap_overlap=fmt(row.get("codebook_remap_overlap")),
                codebook_remap_jaccard=fmt(row.get("codebook_remap_jaccard")),
                residual_codebook_entropy=fmt(row.get("residual_codebook_entropy")),
                residual_codebook_collision_rate=fmt(row.get("residual_codebook_collision_rate")),
                residual_dead_code_rate=fmt(row.get("residual_dead_code_rate")),
                residual_codebook_top_margin=fmt(row.get("residual_codebook_top_margin")),
                residual_slot_code_entropy=fmt(row.get("residual_slot_code_entropy")),
                residual_slot_code_collision_rate=fmt(row.get("residual_slot_code_collision_rate")),
                residual_dead_slot_code_rate=fmt(row.get("residual_dead_slot_code_rate")),
                residual_slot_code_top_margin=fmt(row.get("residual_slot_code_top_margin")),
                residual_codebook_recon_mse=fmt(row.get("residual_codebook_recon_mse")),
                residual_codebook_recon_cosine=fmt(row.get("residual_codebook_recon_cosine")),
                residual_slot_remap_recon_mse=fmt(row.get("residual_slot_remap_recon_mse")),
                residual_slot_remap_recon_cosine=fmt(row.get("residual_slot_remap_recon_cosine")),
                residual_codebook_support_mean=fmt(row.get("residual_codebook_support_mean")),
                residual_codebook_remap_overlap=fmt(row.get("residual_codebook_remap_overlap")),
                residual_codebook_remap_jaccard=fmt(row.get("residual_codebook_remap_jaccard")),
                residual_query_energy_ratio=fmt(row.get("residual_query_energy_ratio")),
                residual_slot_energy_ratio=fmt(row.get("residual_slot_energy_ratio")),
                residual_gate=fmt(row.get("residual_gate")),
                protected_channels=fmt(row.get("protected_channels")),
                protected_channel_fraction=fmt(row.get("protected_channel_fraction")),
                protected_residual_codebook_entropy=fmt(row.get("protected_residual_codebook_entropy")),
                protected_residual_codebook_collision_rate=fmt(
                    row.get("protected_residual_codebook_collision_rate")
                ),
                protected_query_energy_ratio=fmt(row.get("protected_query_energy_ratio")),
                protected_slot_energy_ratio=fmt(row.get("protected_slot_energy_ratio")),
                protected_gate=fmt(row.get("protected_gate")),
                gauge_basis_orthogonality_error=fmt(row.get("gauge_basis_orthogonality_error")),
                gauge_protected_energy_fraction=fmt(row.get("gauge_protected_energy_fraction")),
                gauge_eigenvalue_top_margin=fmt(row.get("gauge_eigenvalue_top_margin")),
                gauge_selected_channels=fmt(row.get("gauge_selected_channels")),
                signal_basis_orthogonality_error=fmt(row.get("signal_basis_orthogonality_error")),
                signal_task_energy_fraction=fmt(row.get("signal_task_energy_fraction")),
                signal_eigenvalue_top_margin=fmt(row.get("signal_eigenvalue_top_margin")),
                signal_selected_channels=fmt(row.get("signal_selected_channels")),
                signal_variance_alignment=fmt(row.get("signal_variance_alignment")),
                signal_query_energy_ratio=fmt(row.get("signal_query_energy_ratio")),
                signal_slot_energy_ratio=fmt(row.get("signal_slot_energy_ratio")),
                signal_gate=fmt(row.get("signal_gate")),
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
    parser.add_argument("--kv-route-budget", type=int, default=2)
    parser.add_argument("--kv-value-budget", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=4)
    parser.add_argument("--residual-codebook-size", type=int, default=4)
    parser.add_argument("--protected-channels", type=int, default=2)
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
        kv_route_budget=args.kv_route_budget,
        kv_value_budget=args.kv_value_budget,
        codebook_size=args.codebook_size,
        residual_codebook_size=args.residual_codebook_size,
        protected_channels=args.protected_channels,
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
