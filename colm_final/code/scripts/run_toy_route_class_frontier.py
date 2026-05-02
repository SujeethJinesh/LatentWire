#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_toy_hub_sticky_frontier_stack as stack


METHODS: tuple[str, ...] = (
    "conditional_prior_base",
    "conditional_prior_quant_error_frontier",
    "conditional_prior_route_class_patch_protect",
    "conditional_prior_route_class_patch_frontier",
    "conditional_prior_route_class_patch_frontier_stop",
    "oracle_base",
    "oracle_quant_error_frontier",
    "oracle_route_class_patch_protect",
    "oracle_route_class_patch_frontier",
    "oracle_route_class_patch_frontier_stop",
)


ROW_KEY_ORDER: tuple[str, ...] = (
    "method",
    "router",
    "frontier_policy",
    "stop_policy",
    "seed",
    "accuracy",
    "mse",
    "accuracy_delta_vs_raw_pairwise",
    "route_accuracy",
    "patch_rank_correlation",
    "protected_oracle_preservation_rate",
    "selected_atom_count",
    "protected_atom_count",
    "average_stop_steps",
    "over_refinement_rate",
    "bytes_proxy",
    "compute_proxy",
    "help_vs_raw_pairwise",
    "harm_vs_raw_pairwise",
)


def _router_method(name: str) -> str:
    return "hub_dictionary_only" if name.startswith("conditional_prior") else "oracle_router_control"


def _router_label(name: str) -> str:
    return "conditional_prior" if name.startswith("conditional_prior") else "oracle_router"


def _route_examples(
    *,
    router_method: str,
    examples: dict[str, torch.Tensor],
    problem: dict[str, torch.Tensor],
    fitted: dict[str, Any],
    config: stack.ToyHubStickyFrontierStackConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    route, _, route_confidence, _ = stack._route_policy(
        router_method,
        clean_source=examples["source"],
        perturbed_source=examples["perturb_source"],
        source_family=examples["source_family"],
        target_family=examples["target_family"],
        problem=problem,
        fitted=fitted,
        config=config,
    )
    source_hub = torch.stack(
        [
            stack._apply_linear(
                examples["source"][idx : idx + 1],
                fitted["source_encoders"][int(examples["source_family"][idx].item())],
            ).squeeze(0)
            for idx in range(examples["source"].shape[0])
        ],
        dim=0,
    )
    atom_scores = torch.softmax(source_hub @ problem["shared_atoms"].T / float(config.route_temperature), dim=-1)
    base_recon = stack._reconstruct_from_hub(
        source_hub,
        route,
        atom_scores,
        torch.ones_like(atom_scores, dtype=torch.bool),
        torch.zeros_like(atom_scores, dtype=torch.bool),
        fitted,
        problem,
        config,
    )
    return route, route_confidence, source_hub, atom_scores


def _exact_patch_scores_for_example(
    *,
    source_hub: torch.Tensor,
    route_id: int,
    atom_score: torch.Tensor,
    target: torch.Tensor,
    fitted: dict[str, Any],
    problem: dict[str, torch.Tensor],
    config: stack.ToyHubStickyFrontierStackConfig,
) -> torch.Tensor:
    keep_all = torch.ones(atom_score.shape[0], dtype=torch.bool)
    protect_none = torch.zeros(atom_score.shape[0], dtype=torch.bool)
    base_recon = stack._reconstruct_from_hub(
        source_hub.view(1, -1),
        torch.tensor([route_id], dtype=torch.long),
        atom_score.view(1, -1),
        keep_all.view(1, -1),
        protect_none.view(1, -1),
        fitted,
        problem,
        config,
    ).squeeze(0)
    base_mse = float((base_recon - target).pow(2).mean().item())
    patch_scores = torch.full((atom_score.shape[0],), float("-inf"), dtype=torch.float32)
    for atom in range(atom_score.shape[0]):
        protect_one = protect_none.clone()
        protect_one[atom] = True
        recon = stack._reconstruct_from_hub(
            source_hub.view(1, -1),
            torch.tensor([route_id], dtype=torch.long),
            atom_score.view(1, -1),
            keep_all.view(1, -1),
            protect_one.view(1, -1),
            fitted,
            problem,
            config,
        ).squeeze(0)
        patch_scores[atom] = base_mse - float((recon - target).pow(2).mean().item())
    return patch_scores


def _fit_route_class_patch_scores(
    *,
    router_method: str,
    calibration: dict[str, torch.Tensor],
    problem: dict[str, torch.Tensor],
    fitted: dict[str, Any],
    config: stack.ToyHubStickyFrontierStackConfig,
) -> torch.Tensor:
    route, _, source_hub, atom_scores = _route_examples(
        router_method=router_method,
        examples=calibration,
        problem=problem,
        fitted=fitted,
        config=config,
    )
    sums = torch.zeros(config.families, config.classes, config.atoms, dtype=torch.float32)
    counts = torch.zeros(config.families, config.classes, dtype=torch.float32)
    for idx in range(calibration["source"].shape[0]):
        route_id = int(route[idx].item())
        class_id = int(calibration["target_label"][idx].item())
        patch_scores = _exact_patch_scores_for_example(
            source_hub=source_hub[idx],
            route_id=route_id,
            atom_score=atom_scores[idx],
            target=calibration["target"][idx],
            fitted=fitted,
            problem=problem,
            config=config,
        )
        sums[route_id, class_id] += patch_scores
        counts[route_id, class_id] += 1.0
    global_default = sums.sum(dim=(0, 1)) / counts.sum().clamp_min(1.0)
    route_default = sums.sum(dim=1) / counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    route_class_scores = torch.zeros_like(sums)
    for route_id in range(config.families):
        for class_id in range(config.classes):
            if float(counts[route_id, class_id].item()) > 0:
                route_class_scores[route_id, class_id] = sums[route_id, class_id] / counts[route_id, class_id]
            elif float(counts[route_id].sum().item()) > 0:
                route_class_scores[route_id, class_id] = route_default[route_id]
            else:
                route_class_scores[route_id, class_id] = global_default
    return route_class_scores


def _select_route_class_masks(
    *,
    route_class_scores: torch.Tensor,
    route: torch.Tensor,
    predicted_class: torch.Tensor,
    atom_scores: torch.Tensor,
    config: stack.ToyHubStickyFrontierStackConfig,
    protect_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep_masks = []
    protected_masks = []
    keep_k = max(1, int(math.ceil(config.keep_fraction * config.atoms)))
    for idx in range(route.shape[0]):
        route_id = int(route[idx].item())
        class_id = int(predicted_class[idx].item())
        score = route_class_scores[route_id, class_id].clamp_min(0.0) * atom_scores[idx]
        if protect_only:
            keep_mask = torch.ones(config.atoms, dtype=torch.bool)
        else:
            keep_mask = stack._topk_mask(score, keep_k)
        protect_mask = stack._topk_mask(score, min(config.protected_atoms, int(keep_mask.sum().item())), candidate_mask=keep_mask)
        keep_masks.append(keep_mask)
        protected_masks.append(protect_mask)
    return torch.stack(keep_masks, dim=0), torch.stack(protected_masks, dim=0)


def _fit_route_class_stop_steps(
    *,
    router_method: str,
    calibration: dict[str, torch.Tensor],
    route_class_scores: torch.Tensor,
    problem: dict[str, torch.Tensor],
    fitted: dict[str, Any],
    config: stack.ToyHubStickyFrontierStackConfig,
) -> torch.Tensor:
    route, _, source_hub, atom_scores = _route_examples(
        router_method=router_method,
        examples=calibration,
        problem=problem,
        fitted=fitted,
        config=config,
    )
    base_recon = stack._reconstruct_from_hub(
        source_hub,
        route,
        atom_scores,
        torch.ones_like(atom_scores, dtype=torch.bool),
        torch.zeros_like(atom_scores, dtype=torch.bool),
        fitted,
        problem,
        config,
    )
    predicted_class = (base_recon @ problem["target_projection"].T).argmax(dim=-1)
    keep_mask, protected_mask = _select_route_class_masks(
        route_class_scores=route_class_scores,
        route=route,
        predicted_class=predicted_class,
        atom_scores=atom_scores,
        config=config,
        protect_only=False,
    )
    best_step_counts = torch.zeros(config.families, config.classes, config.max_steps, dtype=torch.long)
    for idx in range(calibration["source"].shape[0]):
        trajectory = stack._step_trajectory(
            source_hub[idx : idx + 1],
            route[idx : idx + 1],
            atom_scores[idx : idx + 1],
            keep_mask[idx : idx + 1],
            protected_mask[idx : idx + 1],
            fitted,
            problem,
            config,
        )
        step_mse = torch.stack([(step - calibration["target"][idx : idx + 1]).pow(2).mean(dim=-1) for step in trajectory], dim=1)
        best_step = int(step_mse.argmin(dim=1).item())
        best_step_counts[int(route[idx].item()), int(calibration["target_label"][idx].item()), best_step] += 1
    mode_step = torch.zeros(config.families, config.classes, dtype=torch.long)
    for route_id in range(config.families):
        route_counts = best_step_counts[route_id].sum(dim=0)
        global_fallback = int(route_counts.argmax().item())
        for class_id in range(config.classes):
            counts = best_step_counts[route_id, class_id]
            if int(counts.sum().item()) > 0:
                mode_step[route_id, class_id] = int(counts.argmax().item())
            else:
                mode_step[route_id, class_id] = global_fallback
    return mode_step


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


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2 or right.numel() < 2:
        return 0.0
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denom = left_centered.norm() * right_centered.norm()
    if float(denom.item()) <= 1e-8:
        return 0.0
    return float((left_centered @ right_centered / denom).clamp(-1.0, 1.0).item())


def _mask_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = int(right.sum().item())
    if denom == 0:
        return 1.0 if int(left.sum().item()) == 0 else 0.0
    return float((left & right).float().sum().item() / float(denom))


def _route_label_accuracy(route: torch.Tensor, target_family: torch.Tensor) -> float:
    return float(route.eq(target_family).float().mean().item())


def run_experiment(config: stack.ToyHubStickyFrontierStackConfig) -> dict[str, Any]:
    problem, calibration, test = stack._build_dataset(config)
    fitted = stack._fit_problem_components(config, problem, calibration)

    raw_pairwise = torch.stack(
        [
            stack._apply_linear(
                test["source"][idx : idx + 1],
                fitted["pairwise_bridges"][(int(test["source_family"][idx].item()), int(test["target_family"][idx].item()))],
            ).squeeze(0)
            for idx in range(test["source"].shape[0])
        ],
        dim=0,
    )
    raw_pairwise_accuracy = float((raw_pairwise @ problem["target_projection"].T).argmax(dim=-1).eq(test["target_label"]).float().mean().item())
    raw_pairwise_mse = float((raw_pairwise - test["target"]).pow(2).mean().item())

    route_class_scores = {
        router: _fit_route_class_patch_scores(
            router_method=router,
            calibration=calibration,
            problem=problem,
            fitted=fitted,
            config=config,
        )
        for router in {"hub_dictionary_only", "oracle_router_control"}
    }
    route_class_stop = {
        router: _fit_route_class_stop_steps(
            router_method=router,
            calibration=calibration,
            route_class_scores=route_class_scores[router],
            problem=problem,
            fitted=fitted,
            config=config,
        )
        for router in {"hub_dictionary_only", "oracle_router_control"}
    }

    rows = []
    summary_by_router: dict[str, dict[str, float]] = defaultdict(dict)

    for method in METHODS:
        router_method = _router_method(method)
        router_label = _router_label(method)
        route, route_confidence, source_hub, atom_scores = _route_examples(
            router_method=router_method,
            examples=test,
            problem=problem,
            fitted=fitted,
            config=config,
        )
        base_recon = stack._reconstruct_from_hub(
            source_hub,
            route,
            atom_scores,
            torch.ones_like(atom_scores, dtype=torch.bool),
            torch.zeros_like(atom_scores, dtype=torch.bool),
            fitted,
            problem,
            config,
        )
        predicted_class = (base_recon @ problem["target_projection"].T).argmax(dim=-1)

        if method.endswith("_base"):
            keep_mask = torch.ones_like(atom_scores, dtype=torch.bool)
            protected_mask = torch.zeros_like(atom_scores, dtype=torch.bool)
            trajectory = [base_recon]
            selected_step = torch.zeros(test["source"].shape[0], dtype=torch.long)
            frontier_policy = "all_low_bit"
            stop_policy = "none"
        elif method.endswith("_quant_error_frontier"):
            keep_mask, protected_mask, _ = stack._frontier_selection(
                atom_scores,
                route_confidence,
                problem,
                fitted,
                protected_atoms=config.protected_atoms,
                low_bits=config.low_bits,
                high_bits=config.high_bits,
            )
            recon = stack._reconstruct_from_hub(
                source_hub,
                route,
                atom_scores,
                keep_mask,
                protected_mask,
                fitted,
                problem,
                config,
            )
            trajectory = [recon]
            selected_step = torch.zeros(test["source"].shape[0], dtype=torch.long)
            frontier_policy = "quant_error_frontier"
            stop_policy = "none"
        elif method.endswith("_route_class_patch_protect"):
            keep_mask, protected_mask = _select_route_class_masks(
                route_class_scores=route_class_scores[router_method],
                route=route,
                predicted_class=predicted_class,
                atom_scores=atom_scores,
                config=config,
                protect_only=True,
            )
            recon = stack._reconstruct_from_hub(
                source_hub,
                route,
                atom_scores,
                keep_mask,
                protected_mask,
                fitted,
                problem,
                config,
            )
            trajectory = [recon]
            selected_step = torch.zeros(test["source"].shape[0], dtype=torch.long)
            frontier_policy = "route_class_patch_protect"
            stop_policy = "none"
        elif method.endswith("_route_class_patch_frontier"):
            keep_mask, protected_mask = _select_route_class_masks(
                route_class_scores=route_class_scores[router_method],
                route=route,
                predicted_class=predicted_class,
                atom_scores=atom_scores,
                config=config,
                protect_only=False,
            )
            recon = stack._reconstruct_from_hub(
                source_hub,
                route,
                atom_scores,
                keep_mask,
                protected_mask,
                fitted,
                problem,
                config,
            )
            trajectory = [recon]
            selected_step = torch.zeros(test["source"].shape[0], dtype=torch.long)
            frontier_policy = "route_class_patch_frontier"
            stop_policy = "none"
        elif method.endswith("_route_class_patch_frontier_stop"):
            keep_mask, protected_mask = _select_route_class_masks(
                route_class_scores=route_class_scores[router_method],
                route=route,
                predicted_class=predicted_class,
                atom_scores=atom_scores,
                config=config,
                protect_only=False,
            )
            trajectory = stack._step_trajectory(
                source_hub,
                route,
                atom_scores,
                keep_mask,
                protected_mask,
                fitted,
                problem,
                config,
            )
            selected_step = torch.stack(
                [
                    route_class_stop[router_method][int(route[idx].item()), int(predicted_class[idx].item())]
                    for idx in range(route.shape[0])
                ],
                dim=0,
            )
            frontier_policy = "route_class_patch_frontier"
            stop_policy = "route_class_mode_step"
        else:
            raise ValueError(f"Unknown method {method}")

        final = stack._select_by_index(trajectory, selected_step)
        pred_label = (final @ problem["target_projection"].T).argmax(dim=-1)
        accuracy = float(pred_label.eq(test["target_label"]).float().mean().item())
        mse = float((final - test["target"]).pow(2).mean().item())
        step_mse = torch.stack([(step - test["target"]).pow(2).mean(dim=-1) for step in trajectory], dim=1)
        best_step = step_mse.argmin(dim=1)
        over_refinement = float(
            ((selected_step > best_step) & (step_mse.gather(1, selected_step.view(-1, 1)).squeeze(1) > step_mse.min(dim=1).values + 1e-8))
            .float()
            .mean()
            .item()
        )

        patch_correlations = []
        protect_overlap = []
        for idx in range(test["source"].shape[0]):
            exact_patch = _exact_patch_scores_for_example(
                source_hub=source_hub[idx],
                route_id=int(route[idx].item()),
                atom_score=atom_scores[idx],
                target=test["target"][idx],
                fitted=fitted,
                problem=problem,
                config=config,
            )
            route_class_score = route_class_scores[router_method][int(route[idx].item()), int(predicted_class[idx].item())] * atom_scores[idx]
            patch_correlations.append(_spearman(route_class_score, exact_patch))
            oracle_protect = stack._topk_mask(exact_patch, min(config.protected_atoms, config.atoms))
            protect_overlap.append(_mask_overlap(protected_mask[idx], oracle_protect))

        selected_atom_count = float(keep_mask.sum(dim=1).float().mean().item())
        protected_atom_count = float(protected_mask.sum(dim=1).float().mean().item())
        bytes_proxy = float(stack._estimate_hub_bytes(config, int(protected_atom_count), config.families))
        compute_proxy = float(
            stack._estimate_compute_proxy(
                config.dim,
                float(selected_step.float().mean().item()) + 1.0,
                selected_atom_count,
                route_cost=1.10 if router_label == "conditional_prior" else 1.20,
            )
        )
        rows.append(
            {
                "method": method,
                "router": router_label,
                "frontier_policy": frontier_policy,
                "stop_policy": stop_policy,
                "seed": int(config.seed),
                "accuracy": accuracy,
                "mse": mse,
                "accuracy_delta_vs_raw_pairwise": float(accuracy - raw_pairwise_accuracy),
                "route_accuracy": _route_label_accuracy(route, test["target_family"]),
                "patch_rank_correlation": float(sum(patch_correlations) / max(len(patch_correlations), 1)),
                "protected_oracle_preservation_rate": float(sum(protect_overlap) / max(len(protect_overlap), 1)),
                "selected_atom_count": selected_atom_count,
                "protected_atom_count": protected_atom_count,
                "average_stop_steps": float(selected_step.float().mean().item() + 1.0),
                "over_refinement_rate": over_refinement,
                "bytes_proxy": bytes_proxy,
                "compute_proxy": compute_proxy,
                "help_vs_raw_pairwise": float(max(0.0, accuracy - raw_pairwise_accuracy)),
                "harm_vs_raw_pairwise": float(max(0.0, raw_pairwise_accuracy - accuracy)),
            }
        )
        summary_by_router[router_label][frontier_policy + ":" + stop_policy] = accuracy

    interpretation = (
        "This sweep tests whether a route-/class-calibrated patch-effect frontier repairs the negative frontier result from the previous hub sweep. "
        "If route_class_patch variants improve patch-rank correlation and accuracy relative to the quant-error frontier, the next paper lane should replace the current frontier heuristic rather than abandoning protected communication altogether."
    )
    ordered_rows = [{key: row[key] for key in ROW_KEY_ORDER} for row in rows]
    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": ordered_rows,
        "interpretation": interpretation,
        "sources_consulted": [
            "results/query_pool_toy_20260421/protected_frontier_selection_20260421.md",
            "results/query_pool_toy_20260421/hub_router_frontier_sweep_20260421.md",
            "https://arxiv.org/abs/2502.03714",
            "https://arxiv.org/abs/2506.14038",
        ],
    }


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = [
        "# Toy Route-Class Frontier Sweep",
        "",
        "Route-conditioned patch-effect frontier and stop sweep for the hub stack.",
        "",
        "| Method | Router | Frontier | Stop | Accuracy | Delta vs raw | Route acc | Patch corr | Protect-oracle overlap | Selected atoms | Protected atoms | Avg stop steps | Over-refine | Bytes proxy | Compute proxy |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {method} | {router} | {frontier_policy} | {stop_policy} | {accuracy:.4f} | {accuracy_delta_vs_raw_pairwise:.4f} | {route_accuracy:.4f} | {patch_rank_correlation:.4f} | {protected_oracle_preservation_rate:.4f} | {selected_atom_count:.4f} | {protected_atom_count:.4f} | {average_stop_steps:.4f} | {over_refinement_rate:.4f} | {bytes_proxy:.4f} | {compute_proxy:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["interpretation"],
            "",
            "## Sources Consulted",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["sources_consulted"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/query_pool_toy_20260421/route_class_frontier_sweep_20260421.json"),
    )
    parser.add_argument(
        "--output-md",
        type=pathlib.Path,
        default=pathlib.Path("results/query_pool_toy_20260421/route_class_frontier_sweep_20260421.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-examples", type=int, default=192)
    parser.add_argument("--test-examples", type=int, default=192)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--atoms", type=int, default=12)
    parser.add_argument("--families", type=int, default=6)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--source-noise", type=float, default=0.030)
    parser.add_argument("--target-noise", type=float, default=0.026)
    parser.add_argument("--route-noise", type=float, default=0.280)
    parser.add_argument("--perturb-noise", type=float, default=0.550)
    parser.add_argument("--route-code-strength", type=float, default=0.72)
    parser.add_argument("--source-style-strength", type=float, default=0.26)
    parser.add_argument("--target-style-strength", type=float, default=0.22)
    parser.add_argument("--family-scale-jitter", type=float, default=0.16)
    parser.add_argument("--family-bias-scale", type=float, default=0.55)
    parser.add_argument("--hub-snap-strength", type=float, default=0.35)
    parser.add_argument("--route-temperature", type=float, default=0.80)
    parser.add_argument("--confidence-temperature", type=float, default=0.70)
    parser.add_argument("--sticky-margin", type=float, default=0.035)
    parser.add_argument("--keep-fraction", type=float, default=0.70)
    parser.add_argument("--low-bits", type=int, default=3)
    parser.add_argument("--high-bits", type=int, default=8)
    parser.add_argument("--protected-atoms", type=int, default=4)
    parser.add_argument("--verifier-noise", type=float, default=0.07)
    parser.add_argument("--verifier-harm-margin", type=float, default=0.012)
    parser.add_argument("--verifier-stop-threshold", type=float, default=0.85)
    parser.add_argument("--max-steps", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = stack.ToyHubStickyFrontierStackConfig(
        seed=args.seed,
        calibration_examples=args.calibration_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        atoms=args.atoms,
        families=args.families,
        classes=args.classes,
        source_noise=args.source_noise,
        target_noise=args.target_noise,
        route_noise=args.route_noise,
        perturb_noise=args.perturb_noise,
        route_code_strength=args.route_code_strength,
        source_style_strength=args.source_style_strength,
        target_style_strength=args.target_style_strength,
        family_scale_jitter=args.family_scale_jitter,
        family_bias_scale=args.family_bias_scale,
        hub_snap_strength=args.hub_snap_strength,
        route_temperature=args.route_temperature,
        confidence_temperature=args.confidence_temperature,
        sticky_margin=args.sticky_margin,
        keep_fraction=args.keep_fraction,
        low_bits=args.low_bits,
        high_bits=args.high_bits,
        protected_atoms=args.protected_atoms,
        verifier_noise=args.verifier_noise,
        verifier_harm_margin=args.verifier_harm_margin,
        verifier_stop_threshold=args.verifier_stop_threshold,
        max_steps=args.max_steps,
    )
    payload = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_summary(payload, args.output_md)
    return payload


if __name__ == "__main__":
    main()
