#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import asdict
from typing import Any, Sequence

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_toy_hub_sticky_frontier_stack as stack


ROUTER_METHODS: tuple[str, ...] = (
    "hub_dictionary_only",
    "hub_feature_router",
    "hub_sticky_router",
    "confidence_router_control",
    "random_router_control",
    "oracle_router_control",
)


ROUTER_LABELS: dict[str, str] = {
    "hub_dictionary_only": "conditional_prior",
    "hub_feature_router": "feature_router",
    "hub_sticky_router": "sticky_router",
    "confidence_router_control": "confidence_router",
    "random_router_control": "random_router",
    "oracle_router_control": "oracle_router",
}


ROW_KEY_ORDER: tuple[str, ...] = (
    "method",
    "router",
    "frontier",
    "verifier_stop",
    "route_policy",
    "seed",
    "accuracy",
    "mse",
    "accuracy_delta_vs_raw_pairwise",
    "mse_delta_vs_raw_pairwise",
    "route_accuracy",
    "perturbation_stability",
    "atom_recovery",
    "average_stop_steps",
    "over_refinement_rate",
    "bytes_proxy",
    "compute_proxy",
    "frontier_delta_vs_same_router_base",
    "stop_delta_vs_same_router_frontier",
    "help_vs_raw_pairwise",
    "harm_vs_raw_pairwise",
)


def _variant_method(router: str, *, frontier: bool, verifier_stop: bool) -> str:
    suffix = "base"
    if frontier:
        suffix = "frontier"
    if verifier_stop:
        suffix = "frontier_stop"
    return f"{router}_{suffix}"


def _raw_pairwise_row(
    config: stack.ToyHubStickyFrontierStackConfig,
    problem: dict[str, torch.Tensor],
    fitted: dict[str, Any],
    test: dict[str, torch.Tensor],
) -> dict[str, Any]:
    source = test["source"]
    source_family = test["source_family"]
    target_family = test["target_family"]
    recon = torch.stack(
        [
            stack._apply_linear(
                source[idx : idx + 1],
                fitted["pairwise_bridges"][(int(source_family[idx].item()), int(target_family[idx].item()))],
            ).squeeze(0)
            for idx in range(source.shape[0])
        ],
        dim=0,
    )
    metrics = stack._metrics_for_method(
        method="raw_pairwise_bridge",
        route_policy="oracle_pair_route",
        route=target_family.clone(),
        route_prob=torch.nn.functional.one_hot(target_family, num_classes=config.families).float(),
        route_confidence=torch.ones_like(target_family, dtype=torch.float32),
        route_stability=1.0,
        hub_hat=test["hub"],
        atom_scores=torch.zeros(source.shape[0], config.atoms, dtype=torch.float32),
        keep_mask=None,
        protected_mask=None,
        trajectory=[recon],
        selected_step=torch.zeros(source.shape[0], dtype=torch.long),
        stop_reasons={"direct_bridge": int(source.shape[0])},
        problem=problem,
        test=test,
        config=config,
        baseline_accuracy=float((recon @ problem["target_projection"].T).argmax(dim=-1).eq(test["target_label"]).float().mean().item()),
        baseline_mse=float((recon - test["target"]).pow(2).mean().item()),
        bytes_proxy=stack._estimate_bytes_for_bridge(config.families * max(config.families - 1, 1), config.dim),
        compute_proxy=stack._estimate_compute_proxy(config.dim, 1.0, 0.0, route_cost=1.20),
        bit_histogram={"16": int(config.atoms * source.shape[0])},
    )
    row = {
        "method": metrics["method"],
        "router": "pairwise_control",
        "frontier": False,
        "verifier_stop": False,
        "route_policy": metrics["route_policy"],
        "seed": metrics["seed"],
        "accuracy": metrics["accuracy"],
        "mse": metrics["mse"],
        "accuracy_delta_vs_raw_pairwise": metrics["accuracy_delta_vs_raw_pairwise"],
        "mse_delta_vs_raw_pairwise": metrics["mse_delta_vs_raw_pairwise"],
        "route_accuracy": metrics["route_accuracy"],
        "perturbation_stability": metrics["perturbation_stability"],
        "atom_recovery": metrics["atom_recovery"],
        "average_stop_steps": metrics["average_stop_steps"],
        "over_refinement_rate": metrics["over_refinement_rate"],
        "bytes_proxy": metrics["bytes_proxy"],
        "compute_proxy": metrics["compute_proxy"],
        "frontier_delta_vs_same_router_base": 0.0,
        "stop_delta_vs_same_router_frontier": 0.0,
        "help_vs_raw_pairwise": metrics["help_vs_raw_pairwise"],
        "harm_vs_raw_pairwise": metrics["harm_vs_raw_pairwise"],
    }
    return row


def _evaluate_router_variant(
    *,
    router_method: str,
    use_frontier: bool,
    use_verifier_stop: bool,
    config: stack.ToyHubStickyFrontierStackConfig,
    problem: dict[str, torch.Tensor],
    fitted: dict[str, Any],
    test: dict[str, torch.Tensor],
    baseline_accuracy: float,
    baseline_mse: float,
) -> dict[str, Any]:
    source = test["source"]
    perturbed = test["perturb_source"]
    source_family = test["source_family"]
    target_family = test["target_family"]

    route, route_prob, route_confidence, route_policy = stack._route_policy(
        router_method,
        clean_source=source,
        perturbed_source=perturbed,
        source_family=source_family,
        target_family=target_family,
        problem=problem,
        fitted=fitted,
        config=config,
    )
    pert_route, _, _, _ = stack._route_policy(
        router_method,
        clean_source=perturbed,
        perturbed_source=source,
        source_family=source_family,
        target_family=target_family,
        problem=problem,
        fitted=fitted,
        config=config,
    )
    route_stability = float((route == pert_route).float().mean().item())

    source_hub = torch.stack(
        [
            stack._apply_linear(source[idx : idx + 1], fitted["source_encoders"][int(source_family[idx].item())]).squeeze(0)
            for idx in range(source.shape[0])
        ],
        dim=0,
    )
    atom_scores = torch.softmax(source_hub @ problem["shared_atoms"].T / float(config.route_temperature), dim=-1)

    keep_mask = torch.ones_like(atom_scores, dtype=torch.bool)
    protected_mask = torch.zeros_like(atom_scores, dtype=torch.bool)
    bit_histogram: dict[str, int] = {str(config.low_bits): int(config.atoms * source.shape[0])}
    selected_atom_count = 0.0
    protected_atom_count = 0.0

    if use_frontier:
        keep_mask, protected_mask, histogram_counts = stack._frontier_selection(
            atom_scores,
            route_confidence,
            problem,
            fitted,
            protected_atoms=config.protected_atoms,
            low_bits=config.low_bits,
            high_bits=config.high_bits,
        )
        bit_histogram = {
            str(config.low_bits): int(histogram_counts[0].item()),
            str(config.high_bits): int(histogram_counts[1].item()),
        }
        selected_atom_count = float(keep_mask.sum(dim=-1).float().mean().item())
        protected_atom_count = float(protected_mask.sum(dim=-1).float().mean().item())

    base_recon = stack._reconstruct_from_hub(
        source_hub,
        route,
        atom_scores,
        keep_mask,
        protected_mask,
        fitted,
        problem,
        config,
    )
    stop_reasons: dict[str, int]
    selected_step: torch.Tensor
    trajectory: list[torch.Tensor]
    bytes_proxy = stack._estimate_hub_bytes(config, int(protected_atom_count), config.families)
    compute_proxy = stack._estimate_compute_proxy(
        config.dim,
        1.0,
        float(selected_atom_count),
        route_cost=1.22 if use_frontier else 1.08,
    )

    if use_frontier and use_verifier_stop:
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
        score_steps = stack._verifier_scores(trajectory, fitted, problem, config)
        selected_step, stop_reasons = stack._stop_indices(score_steps, config)
        compute_proxy = stack._estimate_compute_proxy(
            config.dim,
            float(selected_step.float().mean().item()) + 1.0,
            float(selected_atom_count),
            route_cost=1.30,
        )
    else:
        trajectory = [base_recon]
        selected_step = torch.zeros(source.shape[0], dtype=torch.long)
        stop_reasons = {"frontier_fixed" if use_frontier else "low_bit_only": int(source.shape[0])}

    metrics = stack._metrics_for_method(
        method=_variant_method(ROUTER_LABELS[router_method], frontier=use_frontier, verifier_stop=use_verifier_stop),
        route_policy=route_policy,
        route=route,
        route_prob=route_prob,
        route_confidence=route_confidence,
        route_stability=route_stability,
        hub_hat=source_hub,
        atom_scores=atom_scores,
        keep_mask=keep_mask,
        protected_mask=protected_mask,
        trajectory=trajectory,
        selected_step=selected_step,
        stop_reasons=stop_reasons,
        problem=problem,
        test=test,
        config=config,
        baseline_accuracy=baseline_accuracy,
        baseline_mse=baseline_mse,
        bytes_proxy=bytes_proxy,
        compute_proxy=compute_proxy,
        bit_histogram=bit_histogram,
    )
    row = {
        "method": metrics["method"],
        "router": ROUTER_LABELS[router_method],
        "frontier": bool(use_frontier),
        "verifier_stop": bool(use_verifier_stop),
        "route_policy": metrics["route_policy"],
        "seed": metrics["seed"],
        "accuracy": metrics["accuracy"],
        "mse": metrics["mse"],
        "accuracy_delta_vs_raw_pairwise": metrics["accuracy_delta_vs_raw_pairwise"],
        "mse_delta_vs_raw_pairwise": metrics["mse_delta_vs_raw_pairwise"],
        "route_accuracy": metrics["route_accuracy"],
        "perturbation_stability": metrics["perturbation_stability"],
        "atom_recovery": metrics["atom_recovery"],
        "average_stop_steps": metrics["average_stop_steps"],
        "over_refinement_rate": metrics["over_refinement_rate"],
        "bytes_proxy": metrics["bytes_proxy"],
        "compute_proxy": metrics["compute_proxy"],
        "frontier_delta_vs_same_router_base": 0.0,
        "stop_delta_vs_same_router_frontier": 0.0,
        "help_vs_raw_pairwise": metrics["help_vs_raw_pairwise"],
        "harm_vs_raw_pairwise": metrics["harm_vs_raw_pairwise"],
    }
    return row


def run_experiment(config: stack.ToyHubStickyFrontierStackConfig) -> dict[str, Any]:
    problem, calibration, test = stack._build_dataset(config)
    fitted = stack._fit_problem_components(config, problem, calibration)

    raw_pairwise = _raw_pairwise_row(config, problem, fitted, test)
    baseline_accuracy = float(raw_pairwise["accuracy"])
    baseline_mse = float(raw_pairwise["mse"])

    rows = [raw_pairwise]
    for router_method in ROUTER_METHODS:
        for use_frontier, use_verifier_stop in ((False, False), (True, False), (True, True)):
            rows.append(
                _evaluate_router_variant(
                    router_method=router_method,
                    use_frontier=use_frontier,
                    use_verifier_stop=use_verifier_stop,
                    config=config,
                    problem=problem,
                    fitted=fitted,
                    test=test,
                    baseline_accuracy=baseline_accuracy,
                    baseline_mse=baseline_mse,
                )
            )

    by_router: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        router = str(row["router"])
        by_router.setdefault(router, {})[str(row["method"])] = row

    router_summary = []
    for router_method in ROUTER_METHODS:
        router = ROUTER_LABELS[router_method]
        base_name = _variant_method(router, frontier=False, verifier_stop=False)
        frontier_name = _variant_method(router, frontier=True, verifier_stop=False)
        stop_name = _variant_method(router, frontier=True, verifier_stop=True)
        base_row = by_router[router][base_name]
        frontier_row = by_router[router][frontier_name]
        stop_row = by_router[router][stop_name]
        frontier_gain = float(frontier_row["accuracy"] - base_row["accuracy"])
        stop_gain = float(stop_row["accuracy"] - frontier_row["accuracy"])
        frontier_row["frontier_delta_vs_same_router_base"] = frontier_gain
        stop_row["frontier_delta_vs_same_router_base"] = frontier_gain
        stop_row["stop_delta_vs_same_router_frontier"] = stop_gain
        router_summary.append(
            {
                "router": router,
                "route_accuracy": float(base_row["route_accuracy"]),
                "base_accuracy": float(base_row["accuracy"]),
                "frontier_accuracy": float(frontier_row["accuracy"]),
                "stop_accuracy": float(stop_row["accuracy"]),
                "frontier_gain": frontier_gain,
                "stop_gain": stop_gain,
                "base_harm": float(base_row["harm_vs_raw_pairwise"]),
                "stop_harm": float(stop_row["harm_vs_raw_pairwise"]),
                "frontier_atom_recovery": float(frontier_row["atom_recovery"]),
                "stop_over_refinement": float(stop_row["over_refinement_rate"]),
                "perturbation_stability": float(base_row["perturbation_stability"]),
            }
        )

    ordered_rows = []
    for row in rows:
        ordered_rows.append({key: row[key] for key in ROW_KEY_ORDER})

    best_non_oracle = max(
        (item for item in router_summary if item["router"] not in {"oracle_router", "random_router"}),
        key=lambda item: item["base_accuracy"],
    )
    oracle_summary = next(item for item in router_summary if item["router"] == "oracle_router")
    best_frontier_gain = max(item["frontier_gain"] for item in router_summary)
    best_stop_gain = max(item["stop_gain"] for item in router_summary)

    interpretation = (
        "This sweep separates the hub base from later frontier and stop heuristics. "
        f"The best non-oracle hub base is {best_non_oracle['router']} at {best_non_oracle['base_accuracy']:.4f}, "
        f"while oracle routing raises the same hub base to {oracle_summary['base_accuracy']:.4f}, above raw pairwise {baseline_accuracy:.4f}. "
        f"However, the current frontier never adds more than {best_frontier_gain:.4f} accuracy and the stop heuristic never adds more than {best_stop_gain:.4f}. "
        "That means route assignment is a real headroom source, but the current quant-error frontier and verifier stop rules are themselves mis-specified and should not be stacked unchanged."
    )

    return {
        "config": asdict(config),
        "rows": ordered_rows,
        "router_summary": router_summary,
        "interpretation": interpretation,
        "sources_consulted": [
            "https://arxiv.org/abs/2502.03714",
            "https://arxiv.org/abs/2506.14038",
            "https://arxiv.org/abs/2311.14125",
            "https://arxiv.org/abs/2204.08396",
        ],
    }


def _format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Toy Hub Router / Frontier Sweep",
        "",
        "Deterministic route-conditioned sweep for hub decoding, protected frontiers, and verifier stopping.",
        "",
        "| Method | Router | Frontier | Stop | Accuracy | Route acc | Stability | Atom recovery | Avg stop steps | Over-refine | Frontier delta | Stop delta | Bytes proxy | Compute proxy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {method} | {router} | {frontier} | {verifier_stop} | {accuracy:.4f} | {route_accuracy:.4f} | {perturbation_stability:.4f} | {atom_recovery:.4f} | {average_stop_steps:.4f} | {over_refinement_rate:.4f} | {frontier_delta_vs_same_router_base:.4f} | {stop_delta_vs_same_router_frontier:.4f} | {bytes_proxy:.4f} | {compute_proxy:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Router Summary",
            "",
            "| Router | Route acc | Stability | Base acc | Frontier acc | Stop acc | Frontier gain | Stop gain | Frontier atom recovery | Stop over-refine |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["router_summary"]:
        lines.append(
            "| {router} | {route_accuracy:.4f} | {perturbation_stability:.4f} | {base_accuracy:.4f} | {frontier_accuracy:.4f} | {stop_accuracy:.4f} | {frontier_gain:.4f} | {stop_gain:.4f} | {frontier_atom_recovery:.4f} | {stop_over_refinement:.4f} |".format(
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
    lines.extend(f"- {source}" for source in payload["sources_consulted"])
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("results/query_pool_toy_20260421/hub_router_frontier_sweep_20260421.json"),
    )
    parser.add_argument(
        "--output-md",
        type=pathlib.Path,
        default=pathlib.Path("results/query_pool_toy_20260421/hub_router_frontier_sweep_20260421.md"),
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
    args.output_md.write_text(_format_markdown(payload), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
