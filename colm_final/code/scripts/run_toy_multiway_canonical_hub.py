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
    "anchor_family_transfer",
    "multiway_gpa_canonical",
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
    "canonical_gap",
    "shared_basis",
    "heldout_pairs_used",
)


@dataclass(frozen=True)
class ToyMultiwayCanonicalHubConfig:
    seed: int = 0
    dim: int = 20
    classes: int = 6
    families: int = 5
    heldout_family: int = 4
    anchor_family: int = 0
    seen_shots_per_class: int = 24
    heldout_shots: tuple[int, ...] = (1, 2, 4, 8)
    test_examples_per_class: int = 64
    class_noise: float = 0.18
    nuisance_rank: int = 4
    nuisance_strength: float = 0.35
    source_noise: float = 0.04
    scale_jitter: float = 0.08
    bias_scale: float = 0.30
    style_strength: float = 0.10
    ridge_lam: float = 1e-2
    gpa_iters: int = 8


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


def _class_centroids(x: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    centroids = []
    default = x.mean(dim=0)
    for class_id in range(num_classes):
        mask = labels.eq(class_id)
        centroids.append(x[mask].mean(dim=0) if bool(mask.any()) else default)
    return torch.stack(centroids, dim=0)


def _build_problem(config: ToyMultiwayCanonicalHubConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    latent_prototypes = _normalize_rows(torch.randn(config.classes, config.dim, generator=gen)) * 3.0
    nuisance_basis = _normalize_rows(torch.randn(config.nuisance_rank, config.dim, generator=gen))
    rotations = torch.stack([_orthogonal_matrix(config.dim, gen) for _ in range(config.families)], dim=0)
    scales = (1.0 + float(config.scale_jitter) * torch.randn(config.families, config.dim, generator=gen)).clamp(0.75, 1.25)
    bias = float(config.bias_scale) * torch.randn(config.families, config.dim, generator=gen)
    style = _normalize_rows(torch.randn(config.families, config.dim, generator=gen)) * float(config.style_strength)
    return {
        "latent_prototypes": latent_prototypes.float(),
        "nuisance_basis": nuisance_basis.float(),
        "rotations": rotations.float(),
        "scales": scales.float(),
        "bias": bias.float(),
        "style": style.float(),
    }


def _sample_family_examples(
    config: ToyMultiwayCanonicalHubConfig,
    problem: dict[str, torch.Tensor],
    *,
    family_id: int,
    shots_per_class: int,
    seed_offset: int,
) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed + seed_offset + family_id * 997)
    labels = torch.arange(config.classes, dtype=torch.long).repeat_interleave(shots_per_class)
    latent = problem["latent_prototypes"][labels].clone()
    latent = latent + float(config.class_noise) * torch.randn(latent.shape, generator=gen, dtype=latent.dtype)
    nuisance_coeff = float(config.nuisance_strength) * torch.randn(
        latent.shape[0],
        config.nuisance_rank,
        generator=gen,
        dtype=latent.dtype,
    )
    latent = latent + nuisance_coeff @ problem["nuisance_basis"]
    latent = latent + 0.08 * torch.roll(latent, shifts=1, dims=-1)
    rotated = latent @ problem["rotations"][family_id]
    styled = rotated + problem["style"][family_id] * torch.tanh(latent)
    source = styled * problem["scales"][family_id] + problem["bias"][family_id]
    source = source + float(config.source_noise) * torch.randn(source.shape, generator=gen, dtype=source.dtype)
    family = torch.full((labels.shape[0],), family_id, dtype=torch.long)
    return {
        "source": source.float(),
        "latent": latent.float(),
        "label": labels.long(),
        "family": family,
    }


def _concat_examples(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "source": torch.cat([item["source"] for item in items], dim=0),
        "latent": torch.cat([item["latent"] for item in items], dim=0),
        "label": torch.cat([item["label"] for item in items], dim=0),
        "family": torch.cat([item["family"] for item in items], dim=0),
    }


def _multiway_gpa(
    centroid_mats: list[torch.Tensor],
    *,
    iters: int,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    canonical = centroid_mats[0]
    transforms = [torch.eye(canonical.shape[1], dtype=canonical.dtype) for _ in centroid_mats]
    for _ in range(max(int(iters), 1)):
        aligned = []
        for index, centroids in enumerate(centroid_mats):
            transform = orthogonal_procrustes(centroids, canonical)
            transforms[index] = transform
            aligned.append(centroids @ transform)
        canonical = torch.stack(aligned, dim=0).mean(dim=0)
    return transforms, canonical


def _centroid_cosine(predicted: torch.Tensor, target: torch.Tensor, labels: torch.Tensor, classes: int) -> float:
    pred_centroids = _class_centroids(predicted, labels, classes)
    tgt_centroids = _class_centroids(target, labels, classes)
    return float(F.cosine_similarity(pred_centroids, tgt_centroids, dim=-1).mean().item())


def _evaluate_predictions(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyMultiwayCanonicalHubConfig,
) -> tuple[float, float, float]:
    logits = predicted @ problem["latent_prototypes"].T
    accuracy = float(logits.argmax(dim=-1).eq(labels).float().mean().item())
    mse = float(F.mse_loss(predicted, target).item())
    centroid_cosine = _centroid_cosine(predicted, target, labels, config.classes)
    return accuracy, mse, centroid_cosine


def _run_methods_for_shot(
    config: ToyMultiwayCanonicalHubConfig,
    problem: dict[str, torch.Tensor],
    *,
    shot: int,
) -> list[dict[str, Any]]:
    seen_family_ids = [family_id for family_id in range(config.families) if family_id != config.heldout_family]
    seen_train_items = [
        _sample_family_examples(
            config,
            problem,
            family_id=family_id,
            shots_per_class=config.seen_shots_per_class,
            seed_offset=11_000,
        )
        for family_id in seen_family_ids
    ]
    seen_train = _concat_examples(seen_train_items)
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
    for item in seen_train_items + [heldout_oracle]:
        family_id = int(item["family"][0].item())
        family_stats[family_id] = _family_center_scale(item["source"])

    seen_train_norm_items = []
    for item in seen_train_items:
        family_id = int(item["family"][0].item())
        center, scale = family_stats[family_id]
        seen_train_norm_items.append(
            {
                **item,
                "source": _standardize(item["source"], center, scale),
            }
        )
    seen_train_norm = _concat_examples(seen_train_norm_items)

    heldout_center, heldout_scale = family_stats[config.heldout_family]
    heldout_fewshot_norm = {
        **heldout_fewshot,
        "source": _standardize(heldout_fewshot["source"], heldout_center, heldout_scale),
    }
    heldout_oracle_norm = {
        **heldout_oracle,
        "source": _standardize(heldout_oracle["source"], heldout_center, heldout_scale),
    }
    heldout_test_norm = {
        **heldout_test,
        "source": _standardize(heldout_test["source"], heldout_center, heldout_scale),
    }

    fewshot_model = _fit_centered_ridge(heldout_fewshot_norm["source"], heldout_fewshot_norm["latent"], config.ridge_lam)
    global_model = _fit_centered_ridge(seen_train_norm["source"], seen_train_norm["latent"], config.ridge_lam)
    oracle_model = _fit_centered_ridge(heldout_oracle_norm["source"], heldout_oracle_norm["latent"], config.ridge_lam)

    anchor_item = seen_train_norm_items[seen_family_ids.index(config.anchor_family)]
    anchor_model = _fit_centered_ridge(anchor_item["source"], anchor_item["latent"], config.ridge_lam)
    anchor_centroids = _class_centroids(anchor_item["source"], anchor_item["label"], config.classes)
    anchor_centroid_mean = anchor_centroids.mean(dim=0)
    heldout_centroids = _class_centroids(heldout_fewshot_norm["source"], heldout_fewshot_norm["label"], config.classes)
    heldout_centroid_mean = heldout_centroids.mean(dim=0)
    anchor_transform = orthogonal_procrustes(heldout_centroids - heldout_centroid_mean, anchor_centroids - anchor_centroid_mean)

    seen_centroid_mats = []
    seen_centroid_means = []
    for item in seen_train_norm_items:
        centroids = _class_centroids(item["source"], item["label"], config.classes)
        centroid_mean = centroids.mean(dim=0)
        seen_centroid_mats.append(centroids - centroid_mean)
        seen_centroid_means.append(centroid_mean)
    gpa_transforms, canonical_centroids = _multiway_gpa(seen_centroid_mats, iters=config.gpa_iters)

    aligned_seen_inputs = []
    aligned_seen_targets = []
    for item, centroid_mean, transform in zip(seen_train_norm_items, seen_centroid_means, gpa_transforms):
        aligned_seen_inputs.append((item["source"] - centroid_mean) @ transform)
        aligned_seen_targets.append(item["latent"])
    canonical_model = _fit_centered_ridge(torch.cat(aligned_seen_inputs, dim=0), torch.cat(aligned_seen_targets, dim=0), config.ridge_lam)

    heldout_canonical_transform = orthogonal_procrustes(
        heldout_centroids - heldout_centroid_mean,
        canonical_centroids,
    )

    predictions: dict[str, torch.Tensor] = {
        "heldout_fewshot_ridge": _predict_centered_ridge(fewshot_model, heldout_test_norm["source"]),
        "global_seen_ridge": _predict_centered_ridge(global_model, heldout_test_norm["source"]),
        "anchor_family_transfer": _predict_centered_ridge(
            anchor_model,
            (heldout_test_norm["source"] - heldout_centroid_mean) @ anchor_transform + anchor_centroid_mean,
        ),
        "multiway_gpa_canonical": _predict_centered_ridge(
            canonical_model,
            (heldout_test_norm["source"] - heldout_centroid_mean) @ heldout_canonical_transform,
        ),
        "oracle_family_ridge": _predict_centered_ridge(oracle_model, heldout_test_norm["source"]),
    }

    aligned_centroids = {
        "heldout_fewshot_ridge": _class_centroids(predictions["heldout_fewshot_ridge"], heldout_test_norm["label"], config.classes),
        "global_seen_ridge": _class_centroids(predictions["global_seen_ridge"], heldout_test_norm["label"], config.classes),
        "anchor_family_transfer": _class_centroids(
            (heldout_test_norm["source"] - heldout_centroid_mean) @ anchor_transform + anchor_centroid_mean,
            heldout_test_norm["label"],
            config.classes,
        ),
        "multiway_gpa_canonical": _class_centroids(
            (heldout_test_norm["source"] - heldout_centroid_mean) @ heldout_canonical_transform,
            heldout_test_norm["label"],
            config.classes,
        ),
        "oracle_family_ridge": _class_centroids(predictions["oracle_family_ridge"], heldout_test_norm["label"], config.classes),
    }

    baseline_pred = predictions["heldout_fewshot_ridge"]
    baseline_accuracy, baseline_mse, _ = _evaluate_predictions(
        predicted=baseline_pred,
        target=heldout_test_norm["latent"],
        labels=heldout_test_norm["label"],
        problem=problem,
        config=config,
    )

    rows = []
    for method in METHODS:
        predicted = predictions[method]
        accuracy, mse, centroid_cosine = _evaluate_predictions(
            predicted=predicted,
            target=heldout_test_norm["latent"],
            labels=heldout_test_norm["label"],
            problem=problem,
            config=config,
        )
        if method == "multiway_gpa_canonical":
            canonical_gap = float(F.mse_loss(aligned_centroids[method], canonical_centroids).item())
            shared_basis = True
        elif method == "anchor_family_transfer":
            canonical_gap = float(F.mse_loss(aligned_centroids[method], anchor_centroids).item())
            shared_basis = True
        else:
            canonical_gap = float(F.mse_loss(aligned_centroids[method], problem["latent_prototypes"]).item())
            shared_basis = method in {"global_seen_ridge", "oracle_family_ridge"}
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
                "canonical_gap": canonical_gap,
                "shared_basis": bool(shared_basis),
                "heldout_pairs_used": int(shot * config.classes if method != "oracle_family_ridge" else config.seen_shots_per_class * config.classes),
            }
        )
    return rows


def run_experiment(config: ToyMultiwayCanonicalHubConfig) -> dict[str, Any]:
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
        best_non_oracle = max(
            non_oracle_rows,
            key=lambda row: (row["accuracy"], -row["mse"]),
        )
        lowest_mse_non_oracle = min(non_oracle_rows, key=lambda row: row["mse"])
        shared_basis_rows = [row for row in non_oracle_rows if row["shared_basis"]]
        best_shared_basis = min(shared_basis_rows, key=lambda row: row["mse"])
        summary[str(int(shot))] = {
            "best_non_oracle_accuracy_method": best_non_oracle["method"],
            "best_non_oracle_accuracy": best_non_oracle["accuracy"],
            "best_non_oracle_accuracy_mse": best_non_oracle["mse"],
            "lowest_mse_non_oracle_method": lowest_mse_non_oracle["method"],
            "lowest_mse_non_oracle": lowest_mse_non_oracle["mse"],
            "best_shared_basis_method": best_shared_basis["method"],
            "best_shared_basis_mse": best_shared_basis["mse"],
            "best_shared_basis_accuracy": best_shared_basis["accuracy"],
        }

    return {
        "config": config_payload,
        "methods": list(METHODS),
        "rows": [{key: row[key] for key in ROW_KEY_ORDER} for row in rows],
        "summary": summary,
    }


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    config = payload["config"]
    summary = payload["summary"]
    lines = [
        "# Toy Multi-Way Canonical Hub Sweep",
        "",
        f"- seed: `{config['seed']}`",
        f"- dim: `{config['dim']}`",
        f"- classes: `{config['classes']}`",
        f"- families: `{config['families']}`",
        f"- held-out family: `{config['heldout_family']}`",
        f"- seen shots / class: `{config['seen_shots_per_class']}`",
        f"- held-out shot grid: `{config['heldout_shots']}`",
        "",
        "This toy isolates whether a multi-way canonical hub helps when one family has only a few paired examples. It is the direct follow-up to the hub-router sweeps: fix the shared basis first, then revisit communication controls.",
        "",
        "References:",
        "- Multi-Way Representation Alignment: https://arxiv.org/abs/2602.06205",
        "- Model Stitching: https://arxiv.org/abs/2303.11277",
        "",
        "| Shot | Method | Accuracy | MSE | dAcc vs few-shot | dMSE vs few-shot | Centroid Cos | Canonical Gap | Shared Basis |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {shot} | {method} | {accuracy:.4f} | {mse:.4f} | {accuracy_delta_vs_fewshot:+.4f} | {mse_delta_vs_fewshot:+.4f} | {centroid_cosine:.4f} | {canonical_gap:.4f} | {shared_basis} |".format(
                **row,
            )
        )
    lines.extend(["", "## Best Non-Oracle by Shot"])
    for shot, item in summary.items():
        lines.append(
            f"- {shot} shot/class: best accuracy `{item['best_non_oracle_accuracy_method']}` ({item['best_non_oracle_accuracy']:.4f} acc, {item['best_non_oracle_accuracy_mse']:.4f} MSE); "
            f"lowest non-oracle MSE `{item['lowest_mse_non_oracle_method']}` ({item['lowest_mse_non_oracle']:.4f}); "
            f"best shared-basis `{item['best_shared_basis_method']}` ({item['best_shared_basis_accuracy']:.4f} acc, {item['best_shared_basis_mse']:.4f} MSE)",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out-family multi-way canonical hub toy.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--output-md", help="Markdown output path. Defaults to the JSON path with a .md suffix.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--families", type=int, default=5)
    parser.add_argument("--heldout-family", type=int, default=4)
    parser.add_argument("--anchor-family", type=int, default=0)
    parser.add_argument("--seen-shots-per-class", type=int, default=24)
    parser.add_argument("--heldout-shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--test-examples-per-class", type=int, default=64)
    parser.add_argument("--class-noise", type=float, default=0.18)
    parser.add_argument("--nuisance-rank", type=int, default=4)
    parser.add_argument("--nuisance-strength", type=float, default=0.35)
    parser.add_argument("--source-noise", type=float, default=0.04)
    parser.add_argument("--scale-jitter", type=float, default=0.08)
    parser.add_argument("--bias-scale", type=float, default=0.30)
    parser.add_argument("--style-strength", type=float, default=0.10)
    parser.add_argument("--ridge-lam", type=float, default=1e-2)
    parser.add_argument("--gpa-iters", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyMultiwayCanonicalHubConfig(
        seed=args.seed,
        dim=args.dim,
        classes=args.classes,
        families=args.families,
        heldout_family=args.heldout_family,
        anchor_family=args.anchor_family,
        seen_shots_per_class=args.seen_shots_per_class,
        heldout_shots=tuple(args.heldout_shots),
        test_examples_per_class=args.test_examples_per_class,
        class_noise=args.class_noise,
        nuisance_rank=args.nuisance_rank,
        nuisance_strength=args.nuisance_strength,
        source_noise=args.source_noise,
        scale_jitter=args.scale_jitter,
        bias_scale=args.bias_scale,
        style_strength=args.style_strength,
        ridge_lam=args.ridge_lam,
        gpa_iters=args.gpa_iters,
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
