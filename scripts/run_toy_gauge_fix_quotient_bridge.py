#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latent_bridge.procrustes import orthogonal_procrustes, ridge_projection


METHODS: tuple[str, ...] = (
    "heldout_fewshot_ridge",
    "global_seen_ridge",
    "gauge_fix_then_bridge",
    "quotient_match_after_fix",
    "oracle_family_ridge",
)


ROW_KEY_ORDER: tuple[str, ...] = (
    "shot",
    "method",
    "seed",
    "heldout_family",
    "accuracy",
    "mse",
    "accuracy_delta_vs_fewshot",
    "mse_delta_vs_fewshot",
    "centroid_cosine",
    "gauge_residual",
    "head_match_accuracy",
    "shared_basis",
    "heldout_pairs_used",
)


@dataclass(frozen=True)
class ToyGaugeFixQuotientBridgeConfig:
    seed: int = 0
    heads: int = 4
    head_dim: int = 6
    classes: int = 5
    families: int = 5
    heldout_family: int = 4
    anchor_family: int = 0
    seen_shots_per_class: int = 24
    heldout_shots: tuple[int, ...] = (1, 2, 4, 8)
    test_examples_per_class: int = 64
    class_noise: float = 0.16
    source_noise: float = 0.03
    nuisance_rank: int = 3
    nuisance_strength: float = 0.25
    head_scale_jitter: float = 0.18
    head_bias_scale: float = 0.18
    ridge_lam: float = 1e-2


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator, dtype=torch.float32))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _fit_centered_ridge(source: torch.Tensor, target: torch.Tensor, ridge_lam: float) -> dict[str, torch.Tensor]:
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    weight = ridge_projection(source - source_mean, target - target_mean, lam=ridge_lam)
    return {
        "source_mean": source_mean,
        "target_mean": target_mean,
        "weight": weight,
    }


def _predict_centered_ridge(model: dict[str, torch.Tensor], source: torch.Tensor) -> torch.Tensor:
    return (source - model["source_mean"]) @ model["weight"] + model["target_mean"]


def _family_center_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-3)
    return center, scale


def _standardize(x: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x - center) / scale


def _flatten_heads(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _unflatten_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    return x.reshape(x.shape[0], heads, head_dim)


def _class_centroids(x: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    default = x.mean(dim=0)
    centroids = []
    for class_id in range(num_classes):
        mask = labels.eq(class_id)
        centroids.append(x[mask].mean(dim=0) if bool(mask.any()) else default)
    return torch.stack(centroids, dim=0)


def _hungarian(cost: torch.Tensor) -> torch.Tensor:
    matrix = cost.detach().cpu().double().tolist()
    n = len(matrix)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = torch.empty(n, dtype=torch.long)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def _build_problem(config: ToyGaugeFixQuotientBridgeConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    prototypes = torch.randn(config.classes, config.heads, config.head_dim, generator=gen, dtype=torch.float32)
    prototypes = _normalize_rows(prototypes.reshape(-1, config.head_dim)).reshape(config.classes, config.heads, config.head_dim)
    prototypes = 3.0 * prototypes
    nuisance_basis = torch.randn(config.nuisance_rank, config.heads, config.head_dim, generator=gen, dtype=torch.float32)
    nuisance_basis = _normalize_rows(nuisance_basis.reshape(-1, config.head_dim)).reshape(config.nuisance_rank, config.heads, config.head_dim)

    permutations = []
    rotations = []
    scales = []
    biases = []
    for family in range(config.families):
        family_gen = _make_generator(config.seed + 10_000 + family * 131)
        permutations.append(torch.randperm(config.heads, generator=family_gen))
        rotations.append(torch.stack([_orthogonal_matrix(config.head_dim, family_gen) for _ in range(config.heads)], dim=0))
        scale = 1.0 + float(config.head_scale_jitter) * torch.randn(
            config.heads,
            config.head_dim,
            generator=family_gen,
            dtype=torch.float32,
        )
        scales.append(scale.clamp(0.65, 1.35))
        biases.append(float(config.head_bias_scale) * torch.randn(config.heads, config.head_dim, generator=family_gen, dtype=torch.float32))

    return {
        "latent_prototypes": prototypes.float(),
        "nuisance_basis": nuisance_basis.float(),
        "permutations": torch.stack(permutations, dim=0).long(),
        "rotations": torch.stack(rotations, dim=0).float(),
        "scales": torch.stack(scales, dim=0).float(),
        "biases": torch.stack(biases, dim=0).float(),
    }


def _apply_family_gauge(
    heads: torch.Tensor,
    *,
    family_id: int,
    problem: dict[str, torch.Tensor],
) -> torch.Tensor:
    perm = problem["permutations"][family_id]
    observed = heads[:, perm].clone()
    rotated = torch.einsum("bhd,hdk->bhk", observed, problem["rotations"][family_id])
    return rotated * problem["scales"][family_id].unsqueeze(0) + problem["biases"][family_id].unsqueeze(0)


def _sample_family_examples(
    config: ToyGaugeFixQuotientBridgeConfig,
    problem: dict[str, torch.Tensor],
    *,
    family_id: int,
    shots_per_class: int,
    seed_offset: int,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset + family_id * 977)
    labels = torch.arange(config.classes, dtype=torch.long).repeat_interleave(shots_per_class)
    latent = problem["latent_prototypes"][labels].clone()
    latent = latent + float(config.class_noise) * torch.randn(latent.shape, generator=gen, dtype=latent.dtype)
    nuisance_coeff = float(config.nuisance_strength) * torch.randn(
        latent.shape[0],
        config.nuisance_rank,
        generator=gen,
        dtype=latent.dtype,
    )
    latent = latent + torch.einsum("br,rhd->bhd", nuisance_coeff, problem["nuisance_basis"])
    source = _apply_family_gauge(latent, family_id=family_id, problem=problem)
    source = source + float(config.source_noise) * torch.randn(source.shape, generator=gen, dtype=source.dtype)
    return {
        "source": _flatten_heads(source).float(),
        "latent": _flatten_heads(latent).float(),
        "label": labels.long(),
        "family": torch.full((labels.shape[0],), family_id, dtype=torch.long),
    }


def _concat_examples(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.cat([item["source"] for item in items], dim=0),
        "latent": torch.cat([item["latent"] for item in items], dim=0),
        "label": torch.cat([item["label"] for item in items], dim=0),
        "family": torch.cat([item["family"] for item in items], dim=0),
    }


def _centroids_by_head(x: torch.Tensor, labels: torch.Tensor, classes: int, heads: int, head_dim: int) -> torch.Tensor:
    head_view = _unflatten_heads(x, heads, head_dim)
    default = head_view.mean(dim=0)
    centroids = []
    for class_id in range(classes):
        mask = labels.eq(class_id)
        centroids.append(head_view[mask].mean(dim=0) if bool(mask.any()) else default)
    return torch.stack(centroids, dim=0)


def _match_family_heads(
    centroids: torch.Tensor,
    anchor_centroids: torch.Tensor,
    *,
    allow_match: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = centroids.shape[1]
    if allow_match:
        cost = torch.empty(heads, heads, dtype=centroids.dtype)
        for src_head in range(heads):
            src_block = centroids[:, src_head, :]
            for anchor_head in range(heads):
                tgt_block = anchor_centroids[:, anchor_head, :]
                rotation = orthogonal_procrustes(src_block, tgt_block)
                aligned = src_block @ rotation
                cost[src_head, anchor_head] = F.mse_loss(aligned, tgt_block)
        assignment = _hungarian(cost)
    else:
        assignment = torch.arange(heads, dtype=torch.long)

    transforms = []
    for src_head in range(heads):
        dst_head = int(assignment[src_head].item())
        transforms.append(orthogonal_procrustes(centroids[:, src_head, :], anchor_centroids[:, dst_head, :]))
    return assignment, torch.stack(transforms, dim=0)


def _canonicalize_examples(
    x: torch.Tensor,
    *,
    assignment: torch.Tensor,
    transforms: torch.Tensor,
    heads: int,
    head_dim: int,
) -> torch.Tensor:
    head_view = _unflatten_heads(x, heads, head_dim)
    aligned = torch.einsum("bhd,hdk->bhk", head_view, transforms)
    canonical = torch.zeros_like(aligned)
    for src_head in range(heads):
        dst_head = int(assignment[src_head].item())
        canonical[:, dst_head, :] = aligned[:, src_head, :]
    return _flatten_heads(canonical)


def _true_assignment_to_anchor(problem: dict[str, torch.Tensor], *, family_id: int, anchor_family: int) -> torch.Tensor:
    family_perm = problem["permutations"][family_id]
    anchor_perm = problem["permutations"][anchor_family]
    assignment = torch.empty_like(family_perm)
    for src_head in range(family_perm.shape[0]):
        canonical_head = int(family_perm[src_head].item())
        anchor_head = int(torch.where(anchor_perm == canonical_head)[0][0].item())
        assignment[src_head] = anchor_head
    return assignment


def _head_match_accuracy(assignment: torch.Tensor, truth: torch.Tensor) -> float:
    return float(assignment.eq(truth).float().mean().item())


def _evaluate_prediction(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyGaugeFixQuotientBridgeConfig,
) -> tuple[float, float, float]:
    prototypes = _flatten_heads(problem["latent_prototypes"])
    logits = predicted @ prototypes.T
    accuracy = float(logits.argmax(dim=-1).eq(labels).float().mean().item())
    mse = float(F.mse_loss(predicted, target).item())
    pred_centroids = _class_centroids(predicted, labels, config.classes)
    tgt_centroids = _class_centroids(target, labels, config.classes)
    centroid_cosine = float(F.cosine_similarity(pred_centroids, tgt_centroids, dim=-1).mean().item())
    return accuracy, mse, centroid_cosine


def _run_methods_for_shot(
    config: ToyGaugeFixQuotientBridgeConfig,
    problem: dict[str, torch.Tensor],
    *,
    shot: int,
) -> list[dict[str, Any]]:
    seen_family_ids = [family for family in range(config.families) if family != config.heldout_family]
    seen_items = [
        _sample_family_examples(
            config,
            problem,
            family_id=family_id,
            shots_per_class=config.seen_shots_per_class,
            seed_offset=11_000,
        )
        for family_id in seen_family_ids
    ]
    heldout_fewshot = _sample_family_examples(
        config,
        problem,
        family_id=config.heldout_family,
        shots_per_class=shot,
        seed_offset=21_000,
    )
    heldout_oracle = _sample_family_examples(
        config,
        problem,
        family_id=config.heldout_family,
        shots_per_class=config.seen_shots_per_class,
        seed_offset=31_000,
    )
    heldout_test = _sample_family_examples(
        config,
        problem,
        family_id=config.heldout_family,
        shots_per_class=config.test_examples_per_class,
        seed_offset=41_000,
    )

    family_stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for item in seen_items + [heldout_oracle]:
        family_id = int(item["family"][0].item())
        family_stats[family_id] = _family_center_scale(item["source"])

    seen_norm_items = []
    for item in seen_items:
        family_id = int(item["family"][0].item())
        center, scale = family_stats[family_id]
        seen_norm_items.append({**item, "source": _standardize(item["source"], center, scale)})
    heldout_center, heldout_scale = family_stats[config.heldout_family]
    heldout_fewshot_norm = {**heldout_fewshot, "source": _standardize(heldout_fewshot["source"], heldout_center, heldout_scale)}
    heldout_oracle_norm = {**heldout_oracle, "source": _standardize(heldout_oracle["source"], heldout_center, heldout_scale)}
    heldout_test_norm = {**heldout_test, "source": _standardize(heldout_test["source"], heldout_center, heldout_scale)}

    seen_train_norm = _concat_examples(seen_norm_items)
    fewshot_model = _fit_centered_ridge(heldout_fewshot_norm["source"], heldout_fewshot_norm["latent"], config.ridge_lam)
    global_model = _fit_centered_ridge(seen_train_norm["source"], seen_train_norm["latent"], config.ridge_lam)
    oracle_model = _fit_centered_ridge(heldout_oracle_norm["source"], heldout_oracle_norm["latent"], config.ridge_lam)

    anchor_item = next(item for item in seen_norm_items if int(item["family"][0].item()) == config.anchor_family)
    anchor_centroids = _centroids_by_head(anchor_item["source"], anchor_item["label"], config.classes, config.heads, config.head_dim)

    canonical_seen_nomatch = []
    canonical_seen_match = []
    canonical_seen_latent = []
    for item in seen_norm_items:
        centroids = _centroids_by_head(item["source"], item["label"], config.classes, config.heads, config.head_dim)
        assignment_nomatch, transforms_nomatch = _match_family_heads(centroids, anchor_centroids, allow_match=False)
        assignment_match, transforms_match = _match_family_heads(centroids, anchor_centroids, allow_match=True)
        canonical_seen_nomatch.append(
            _canonicalize_examples(
                item["source"],
                assignment=assignment_nomatch,
                transforms=transforms_nomatch,
                heads=config.heads,
                head_dim=config.head_dim,
            )
        )
        canonical_seen_match.append(
            _canonicalize_examples(
                item["source"],
                assignment=assignment_match,
                transforms=transforms_match,
                heads=config.heads,
                head_dim=config.head_dim,
            )
        )
        canonical_seen_latent.append(item["latent"])

    canonical_nomatch_model = _fit_centered_ridge(
        torch.cat(canonical_seen_nomatch, dim=0),
        torch.cat(canonical_seen_latent, dim=0),
        config.ridge_lam,
    )
    canonical_match_model = _fit_centered_ridge(
        torch.cat(canonical_seen_match, dim=0),
        torch.cat(canonical_seen_latent, dim=0),
        config.ridge_lam,
    )

    heldout_centroids = _centroids_by_head(
        heldout_fewshot_norm["source"],
        heldout_fewshot_norm["label"],
        config.classes,
        config.heads,
        config.head_dim,
    )
    assignment_nomatch, transforms_nomatch = _match_family_heads(heldout_centroids, anchor_centroids, allow_match=False)
    assignment_match, transforms_match = _match_family_heads(heldout_centroids, anchor_centroids, allow_match=True)
    heldout_nomatch = _canonicalize_examples(
        heldout_test_norm["source"],
        assignment=assignment_nomatch,
        transforms=transforms_nomatch,
        heads=config.heads,
        head_dim=config.head_dim,
    )
    heldout_match = _canonicalize_examples(
        heldout_test_norm["source"],
        assignment=assignment_match,
        transforms=transforms_match,
        heads=config.heads,
        head_dim=config.head_dim,
    )

    predictions: dict[str, torch.Tensor] = {
        "heldout_fewshot_ridge": _predict_centered_ridge(fewshot_model, heldout_test_norm["source"]),
        "global_seen_ridge": _predict_centered_ridge(global_model, heldout_test_norm["source"]),
        "gauge_fix_then_bridge": _predict_centered_ridge(canonical_nomatch_model, heldout_nomatch),
        "quotient_match_after_fix": _predict_centered_ridge(canonical_match_model, heldout_match),
        "oracle_family_ridge": _predict_centered_ridge(oracle_model, heldout_test_norm["source"]),
    }

    baseline_accuracy, baseline_mse, _ = _evaluate_prediction(
        predicted=predictions["heldout_fewshot_ridge"],
        target=heldout_test_norm["latent"],
        labels=heldout_test_norm["label"],
        problem=problem,
        config=config,
    )

    truth_assignment = _true_assignment_to_anchor(problem, family_id=config.heldout_family, anchor_family=config.anchor_family)
    nomatch_centroids = _centroids_by_head(heldout_nomatch, heldout_test_norm["label"], config.classes, config.heads, config.head_dim)
    match_centroids = _centroids_by_head(heldout_match, heldout_test_norm["label"], config.classes, config.heads, config.head_dim)

    rows = []
    for method in METHODS:
        predicted = predictions[method]
        accuracy, mse, centroid_cosine = _evaluate_prediction(
            predicted=predicted,
            target=heldout_test_norm["latent"],
            labels=heldout_test_norm["label"],
            problem=problem,
            config=config,
        )
        if method == "gauge_fix_then_bridge":
            gauge_residual = float(F.mse_loss(nomatch_centroids, anchor_centroids).item())
            head_match_accuracy = _head_match_accuracy(assignment_nomatch, truth_assignment)
            shared_basis = True
        elif method == "quotient_match_after_fix":
            gauge_residual = float(F.mse_loss(match_centroids, anchor_centroids).item())
            head_match_accuracy = _head_match_accuracy(assignment_match, truth_assignment)
            shared_basis = True
        else:
            gauge_residual = float(
                F.mse_loss(
                    _class_centroids(predicted, heldout_test_norm["label"], config.classes),
                    _flatten_heads(problem["latent_prototypes"]),
                ).item()
            )
            head_match_accuracy = None
            shared_basis = False

        rows.append(
            {
                "shot": int(shot),
                "method": method,
                "seed": int(config.seed),
                "heldout_family": int(config.heldout_family),
                "accuracy": accuracy,
                "mse": mse,
                "accuracy_delta_vs_fewshot": accuracy - baseline_accuracy,
                "mse_delta_vs_fewshot": mse - baseline_mse,
                "centroid_cosine": centroid_cosine,
                "gauge_residual": gauge_residual,
                "head_match_accuracy": head_match_accuracy,
                "shared_basis": bool(shared_basis),
                "heldout_pairs_used": int(shot * config.classes if method != "oracle_family_ridge" else config.seen_shots_per_class * config.classes),
            }
        )
    return rows


def run_experiment(config: ToyGaugeFixQuotientBridgeConfig) -> dict[str, Any]:
    problem = _build_problem(config)
    rows = []
    for shot in config.heldout_shots:
        rows.extend(_run_methods_for_shot(config, problem, shot=int(shot)))
    config_payload = asdict(config)
    config_payload["heldout_shots"] = list(config.heldout_shots)
    summary: dict[str, Any] = {}
    for shot in config.heldout_shots:
        shot_rows = [row for row in rows if row["shot"] == int(shot)]
        non_oracle_rows = [row for row in shot_rows if row["method"] != "oracle_family_ridge"]
        best_accuracy = max(non_oracle_rows, key=lambda row: (row["accuracy"], -row["mse"]))
        lowest_mse = min(non_oracle_rows, key=lambda row: row["mse"])
        best_shared = min([row for row in non_oracle_rows if row["shared_basis"]], key=lambda row: row["mse"])
        summary[str(int(shot))] = {
            "best_non_oracle_accuracy_method": best_accuracy["method"],
            "best_non_oracle_accuracy": best_accuracy["accuracy"],
            "best_non_oracle_accuracy_mse": best_accuracy["mse"],
            "lowest_mse_non_oracle_method": lowest_mse["method"],
            "lowest_mse_non_oracle": lowest_mse["mse"],
            "best_shared_basis_method": best_shared["method"],
            "best_shared_basis_mse": best_shared["mse"],
        }
    return {
        "config": config_payload,
        "methods": list(METHODS),
        "rows": [{key: row[key] for key in ROW_KEY_ORDER} for row in rows],
        "summary": summary,
    }


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    config = payload["config"]
    rows = payload["rows"]
    summary = payload["summary"]
    lines = [
        "# Toy Gauge-Fix Quotient Bridge Sweep",
        "",
        f"- seed: `{config['seed']}`",
        f"- heads: `{config['heads']}`",
        f"- head dim: `{config['head_dim']}`",
        f"- classes: `{config['classes']}`",
        f"- families: `{config['families']}`",
        f"- held-out family: `{config['heldout_family']}`",
        f"- anchor family: `{config['anchor_family']}`",
        f"- seen shots / class: `{config['seen_shots_per_class']}`",
        f"- held-out shot grid: `{config['heldout_shots']}`",
        "",
        "This toy isolates whether gauge fixing and quotient-aware head matching help held-out-family transport when the observed source heads differ by family-specific permutations and per-head linear gauges.",
        "",
        "References:",
        "- Complete Characterization of Gauge Symmetries in Transformer Architectures: https://openreview.net/forum?id=KrkbYbK0cH",
        "- GaugeKV: https://openreview.net/forum?id=rSxYPLzyBu",
        "",
        "| Shot | Method | Accuracy | MSE | dAcc vs few-shot | dMSE vs few-shot | Centroid cosine | Gauge residual | Head-match acc | Shared Basis |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "True" if value else "False"
        return f"{float(value):.4f}"

    for row in rows:
        lines.append(
            "| {shot} | {method} | {accuracy} | {mse} | {accuracy_delta_vs_fewshot} | {mse_delta_vs_fewshot} | {centroid_cosine} | {gauge_residual} | {head_match_accuracy} | {shared_basis} |".format(
                **{key: fmt(value) if key != "method" else value for key, value in row.items()}
            )
        )

    lines.extend(["", "## Best Non-Oracle by Shot"])
    for shot, item in summary.items():
        lines.append(
            f"- {shot} shot/class: best accuracy `{item['best_non_oracle_accuracy_method']}` ({item['best_non_oracle_accuracy']:.4f} acc, {item['best_non_oracle_accuracy_mse']:.4f} MSE); "
            f"lowest non-oracle MSE `{item['lowest_mse_non_oracle_method']}` ({item['lowest_mse_non_oracle']:.4f}); "
            f"best shared-basis `{item['best_shared_basis_method']}` ({item['best_shared_basis_mse']:.4f} MSE)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out-family gauge-fix quotient bridge toy.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=6)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--families", type=int, default=5)
    parser.add_argument("--heldout-family", type=int, default=4)
    parser.add_argument("--anchor-family", type=int, default=0)
    parser.add_argument("--seen-shots-per-class", type=int, default=24)
    parser.add_argument("--heldout-shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--test-examples-per-class", type=int, default=64)
    parser.add_argument("--class-noise", type=float, default=0.16)
    parser.add_argument("--source-noise", type=float, default=0.03)
    parser.add_argument("--nuisance-rank", type=int, default=3)
    parser.add_argument("--nuisance-strength", type=float, default=0.25)
    parser.add_argument("--head-scale-jitter", type=float, default=0.18)
    parser.add_argument("--head-bias-scale", type=float, default=0.18)
    parser.add_argument("--ridge-lam", type=float, default=1e-2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyGaugeFixQuotientBridgeConfig(
        seed=args.seed,
        heads=args.heads,
        head_dim=args.head_dim,
        classes=args.classes,
        families=args.families,
        heldout_family=args.heldout_family,
        anchor_family=args.anchor_family,
        seen_shots_per_class=args.seen_shots_per_class,
        heldout_shots=tuple(args.heldout_shots),
        test_examples_per_class=args.test_examples_per_class,
        class_noise=args.class_noise,
        source_noise=args.source_noise,
        nuisance_rank=args.nuisance_rank,
        nuisance_strength=args.nuisance_strength,
        head_scale_jitter=args.head_scale_jitter,
        head_bias_scale=args.head_bias_scale,
        ridge_lam=args.ridge_lam,
    )
    payload = run_experiment(config)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path = pathlib.Path(args.output_md) if args.output_md else output_path.with_suffix(".md")
    write_markdown_summary(payload, markdown_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
