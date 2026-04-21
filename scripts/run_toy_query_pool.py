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

    return {
        "codebook_entropy": float(code_entropy.item()),
        "codebook_collision_rate": float(code_collision_rate.item()),
        "dead_code_rate": float(dead_code_rate.item()),
        "codebook_top_margin": float(code_margin.item()),
        "slot_code_entropy": float(slot_entropy.item()),
        "slot_code_collision_rate": float(slot_collision_rate.item()),
        "dead_slot_code_rate": float(dead_slot_code_rate.item()),
        "slot_code_top_margin": float(slot_margin.item()),
        "codebook_recon_mse": float(query_recon_mse.item()),
        "codebook_recon_cosine": float(query_recon_cosine.item()),
        "slot_remap_recon_mse": float(slot_recon_mse.item()),
        "slot_remap_recon_cosine": float(slot_recon_cosine.item()),
        "codebook_support_mean": float(sum(unique_codes) / max(len(unique_codes), 1)),
        "codebook_remap_overlap": float(sum(overlap_scores) / max(len(overlap_scores), 1)),
        "codebook_remap_jaccard": float(sum(jaccard_scores) / max(len(jaccard_scores), 1)),
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
    return metrics


def run_experiment(config: ToyConfig, scenarios: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = _make_base_tensors(config)
    method_seed_offsets = {
        "topk": 101,
        "query_pool": 211,
        "preconditioned_query_pool": 307,
        "constrained_preconditioned_query_pool": 359,
        "route_atom": 401,
        "asymmetric_kv_budget": 463,
        "codebook_remap": 509,
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
        ]
        for method, factory, margin_k in methods:
            torch.manual_seed(config.seed + scenario_idx * 10_000 + method_seed_offsets[method])
            model = factory()
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
            else:
                budget = config.kv_route_budget + config.kv_value_budget
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "budget": budget,
                    "kv_route_budget": config.kv_route_budget if method == "asymmetric_kv_budget" else None,
                    "kv_value_budget": config.kv_value_budget if method == "asymmetric_kv_budget" else None,
                    "codebook_size": config.codebook_size if method == "codebook_remap" else None,
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
        "| Scenario | Method | Budget | KV route budget | KV value budget | Codebook size | Task acc | Rec MSE | Route entropy | Collision | Dead slots | Top margin | Atom entropy | Atom collision | Dead atoms | Atom margin | Precond cond. | Cosine drift | Norm ratio | Abs scale ratio | KV route entropy | KV value entropy | KV route collision | KV value collision | KV route dead | KV value dead | KV route margin | KV value margin | KV overlap | KV Jaccard | KV KL | KV cosine | KV gap | KV gate mean | KV gate std | Constrained scale min | Constrained scale max | Constrained scale mean | Codebook entropy | Codebook collision | Dead codes | Codebook margin | Slot code entropy | Slot code collision | Dead slot codes | Slot code margin | Codebook recon MSE | Codebook recon cosine | Slot remap recon MSE | Slot remap recon cosine | Codebook support | Remap overlap | Remap Jaccard |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {method} | {budget} | {kv_route_budget} | {kv_value_budget} | {codebook_size} | {task_acc} | {rec_mse} | "
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
            "{codebook_support_mean} | {codebook_remap_overlap} | {codebook_remap_jaccard} |".format(
                scenario=row["scenario"],
                method=row["method"],
                budget=row["budget"],
                kv_route_budget=fmt(row.get("kv_route_budget")),
                kv_value_budget=fmt(row.get("kv_value_budget")),
                codebook_size=fmt(row.get("codebook_size")),
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
