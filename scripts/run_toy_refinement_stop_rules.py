#!/usr/bin/env python3
"""Deterministic toy ablation for target-side refinement stop rules.

The setup reuses the iterative latent refinement toy and asks a narrower
question: given the same candidate refinement trajectory, which stopping rule
prevents useful repair from turning into over-refinement harm?
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_toy_iterative_latent_refinement as base


METHODS: tuple[str, ...] = (
    "fixed_1_step",
    "fixed_2_step",
    "fixed_4_step",
    "confidence_stop",
    "score_drift_stop",
    "verifier_harm_stop",
    "oracle_stop",
)


@dataclass(frozen=True)
class ToyRefinementStopRulesConfig:
    seed: int = 0
    examples: int = 160
    dim: int = 24
    classes: int = 5
    styles: int = 7
    quant_bits: int = 4
    bridge_noise: float = 0.20
    source_noise: float = 0.08
    refinement_rate: float = 0.46
    confidence_threshold: float = 0.88
    score_drift_threshold: float = 0.18
    verifier_harm_margin: float = 0.018
    max_steps: int = 4


def _base_config(config: ToyRefinementStopRulesConfig) -> base.ToyIterativeLatentRefinementConfig:
    return base.ToyIterativeLatentRefinementConfig(
        seed=config.seed,
        examples=config.examples,
        dim=config.dim,
        classes=config.classes,
        styles=config.styles,
        quant_bits=config.quant_bits,
        bridge_noise=config.bridge_noise,
        source_noise=config.source_noise,
        refinement_rate=config.refinement_rate,
    )


def _bytes_for_values(count: int, bits: float) -> float:
    return math.ceil(float(count) * float(bits) / 8.0)


def _base_bytes(config: ToyRefinementStopRulesConfig) -> float:
    return _bytes_for_values(int(config.dim), float(config.quant_bits)) + 4.0


def _select_by_index(trajectory: list[torch.Tensor], stop_index: torch.Tensor) -> torch.Tensor:
    stacked = torch.stack(trajectory, dim=0)
    gather_index = stop_index.view(1, -1, 1).expand(1, -1, stacked.shape[-1])
    return stacked.gather(dim=0, index=gather_index).squeeze(0)


def _confidence_for_steps(trajectory: list[torch.Tensor], problem: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([base._confidence(step, problem) for step in trajectory], dim=1)


def _logit_drift_for_steps(trajectory: list[torch.Tensor], problem: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = [base._logits(step, problem) for step in trajectory]
    drifts = [torch.zeros(logits[0].shape[0], dtype=logits[0].dtype)]
    for previous, current in zip(logits, logits[1:]):
        drifts.append((current - previous).pow(2).mean(dim=-1).sqrt())
    return torch.stack(drifts, dim=1)


def _manifold_residual(latent_hat: torch.Tensor, problem: dict[str, torch.Tensor]) -> torch.Tensor:
    logits = base._logits(latent_hat, problem)
    class_anchor = torch.softmax(logits / 1.3, dim=-1) @ problem["class_prototypes"]
    style_logits = (latent_hat @ problem["style_atoms"].T) / math.sqrt(float(latent_hat.shape[-1]))
    style_anchor = torch.softmax(style_logits / 1.1, dim=-1) @ problem["style_atoms"]
    anchor = class_anchor + 0.52 * style_anchor
    return (latent_hat - anchor).pow(2).mean(dim=-1).sqrt()


def _verifier_score_for_steps(trajectory: list[torch.Tensor], problem: dict[str, torch.Tensor]) -> torch.Tensor:
    scores = []
    for step in trajectory:
        confidence = base._confidence(step, problem)
        residual = _manifold_residual(step, problem)
        logits = base._logits(step, problem)
        top2 = torch.topk(torch.softmax(logits, dim=-1), k=2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        scores.append(confidence + 0.35 * margin - 0.18 * residual)
    return torch.stack(scores, dim=1)


def _first_true_or_last(mask: torch.Tensor, *, last_index: int) -> torch.Tensor:
    examples, steps = mask.shape
    indices = torch.arange(steps, dtype=torch.long).view(1, -1).expand(examples, -1)
    sentinel = torch.full_like(indices, fill_value=int(last_index))
    return torch.where(mask, indices, sentinel).min(dim=1).values


def _fixed_stop_indices(examples: int, step_number: int) -> torch.Tensor:
    return torch.full((examples,), fill_value=int(step_number) - 1, dtype=torch.long)


def _confidence_stop_indices(confidence: torch.Tensor, config: ToyRefinementStopRulesConfig) -> torch.Tensor:
    mask = confidence >= float(config.confidence_threshold)
    return _first_true_or_last(mask, last_index=confidence.shape[1] - 1)


def _score_drift_stop_indices(drift: torch.Tensor, config: ToyRefinementStopRulesConfig) -> torch.Tensor:
    eligible = torch.zeros_like(drift, dtype=torch.bool)
    eligible[:, 1:] = drift[:, 1:] <= float(config.score_drift_threshold)
    return _first_true_or_last(eligible, last_index=drift.shape[1] - 1)


def _verifier_harm_stop_indices(scores: torch.Tensor, config: ToyRefinementStopRulesConfig) -> torch.Tensor:
    drop = scores[:, 1:] < (scores[:, :-1] - float(config.verifier_harm_margin))
    stop_before_drop = torch.zeros_like(scores, dtype=torch.bool)
    stop_before_drop[:, :-1] = drop
    return _first_true_or_last(stop_before_drop, last_index=scores.shape[1] - 1)


def _oracle_stop_indices(trajectory: list[torch.Tensor], problem: dict[str, torch.Tensor]) -> torch.Tensor:
    latent = problem["latent"]
    step_mse = torch.stack([(step - latent).pow(2).mean(dim=-1) for step in trajectory], dim=1)
    logits = [base._logits(step, problem).argmax(dim=-1) for step in trajectory]
    correct = torch.stack([pred == problem["class_target"] for pred in logits], dim=1)
    score = step_mse - 0.005 * correct.float()
    return score.argmin(dim=1)


def _stop_reasons(
    method: str,
    stop_index: torch.Tensor,
    confidence: torch.Tensor,
    drift: torch.Tensor,
    scores: torch.Tensor,
    config: ToyRefinementStopRulesConfig,
) -> dict[str, int]:
    max_index = confidence.shape[1] - 1
    if method.startswith("fixed_"):
        reason = f"fixed_{int(stop_index[0].item()) + 1}_step"
        return {reason: int(stop_index.numel())}
    if method == "confidence_stop":
        stopped = stop_index < max_index
        return {
            "confidence_reached": int(stopped.sum().item()),
            "max_steps": int((~stopped).sum().item()),
        }
    if method == "score_drift_stop":
        stopped = stop_index < max_index
        selected_drift = drift.gather(1, stop_index.view(-1, 1)).squeeze(1)
        return {
            "score_drift_small": int((stopped & (selected_drift <= config.score_drift_threshold)).sum().item()),
            "max_steps": int((~stopped).sum().item()),
        }
    if method == "verifier_harm_stop":
        stopped = stop_index < max_index
        next_index = (stop_index + 1).clamp(max=max_index)
        selected = scores.gather(1, stop_index.view(-1, 1)).squeeze(1)
        next_score = scores.gather(1, next_index.view(-1, 1)).squeeze(1)
        harm_stop = stopped & (next_score < selected - config.verifier_harm_margin)
        return {
            "verifier_predicted_harm": int(harm_stop.sum().item()),
            "max_steps": int((~stopped).sum().item()),
        }
    if method == "oracle_stop":
        return {
            "oracle_best_prefix": int((stop_index < max_index).sum().item()),
            "oracle_max_steps": int((stop_index == max_index).sum().item()),
        }
    raise ValueError(f"unknown stop method: {method}")


def _ece(confidence: torch.Tensor, correct: torch.Tensor, *, bins: int = 5) -> float:
    return base._ece(confidence, correct, bins=bins)


def _metrics_for(
    method: str,
    trajectory: list[torch.Tensor],
    stop_index: torch.Tensor,
    problem: dict[str, torch.Tensor],
    config: ToyRefinementStopRulesConfig,
    confidence_by_step: torch.Tensor,
    drift_by_step: torch.Tensor,
    score_by_step: torch.Tensor,
) -> dict[str, Any]:
    start = trajectory[0]
    final = _select_by_index(trajectory, stop_index)
    logits = base._logits(final, problem)
    confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
    pred = logits.argmax(dim=-1)
    correct = pred == problem["class_target"]
    start_correct = base._logits(start, problem).argmax(dim=-1) == problem["class_target"]
    latent_mse = (final - problem["latent"]).pow(2).mean(dim=-1)
    start_mse = (start - problem["latent"]).pow(2).mean(dim=-1)
    step_mse = torch.stack([(step - problem["latent"]).pow(2).mean(dim=-1) for step in trajectory], dim=1)
    best_step = step_mse[:, : int(config.max_steps)].argmin(dim=1)
    best_step_mse = step_mse[:, : int(config.max_steps)].min(dim=1).values
    average_steps = float((stop_index.float() + 1.0).mean().item())
    extra_step_bytes = _bytes_for_values(int(config.dim), 3.0)
    compute = float(config.dim) + float(config.dim) * max(average_steps - 1.0, 0.0) * 0.38
    bytes_proxy = _base_bytes(config) + max(average_steps - 1.0, 0.0) * extra_step_bytes
    stop_reasons = _stop_reasons(method, stop_index, confidence_by_step, drift_by_step, score_by_step, config)
    reg_mse = F.mse_loss(base._regression_prediction(final, problem), problem["reg_target"])

    return {
        "method": method,
        "task_accuracy": float(correct.float().mean().item()),
        "mse": float(latent_mse.mean().item()),
        "regression_mse": float(reg_mse.item()),
        "average_steps": average_steps,
        "compute_proxy": compute,
        "bytes_proxy": bytes_proxy,
        "help_rate": float(((~start_correct) & correct).float().mean().item()),
        "harm_rate": float((start_correct & ~correct).float().mean().item()),
        "mse_help_rate": float((latent_mse < start_mse - 1e-8).float().mean().item()),
        "mse_harm_rate": float((latent_mse > start_mse + 1e-8).float().mean().item()),
        "over_refinement_rate": float(((stop_index > best_step) & (latent_mse > best_step_mse + 1e-8)).float().mean().item()),
        "mean_confidence": float(confidence.mean().item()),
        "confidence_ece": _ece(confidence, correct),
        "stop_reasons": stop_reasons,
        "stop_histogram": {
            str(index + 1): int((stop_index == index).sum().item()) for index in range(int(config.max_steps))
        },
    }


def run_experiment(config: ToyRefinementStopRulesConfig) -> dict[str, Any]:
    problem = base._make_problem(_base_config(config))
    start = base._initial_bridge(problem, _base_config(config))
    trajectory = base._fixed_refinement(start, problem, _base_config(config), steps=int(config.max_steps))
    confidence_by_step = _confidence_for_steps(trajectory, problem)
    drift_by_step = _logit_drift_for_steps(trajectory, problem)
    score_by_step = _verifier_score_for_steps(trajectory, problem)
    examples = int(config.examples)

    stop_indices = {
        "fixed_1_step": _fixed_stop_indices(examples, 1),
        "fixed_2_step": _fixed_stop_indices(examples, 2),
        "fixed_4_step": _fixed_stop_indices(examples, 4),
        "confidence_stop": _confidence_stop_indices(confidence_by_step, config),
        "score_drift_stop": _score_drift_stop_indices(drift_by_step, config),
        "verifier_harm_stop": _verifier_harm_stop_indices(score_by_step, config),
        "oracle_stop": _oracle_stop_indices(trajectory, problem),
    }
    rows = [
        _metrics_for(
            method,
            trajectory,
            stop_indices[method],
            problem,
            config,
            confidence_by_step,
            drift_by_step,
            score_by_step,
        )
        for method in METHODS
    ]
    return {
        "config": asdict(config),
        "methods": list(METHODS),
        "rows": rows,
        "interpretation": (
            "Stop rules isolate whether target-side repair should halt on confidence, score drift, "
            "verifier-predicted harm, or an oracle best-prefix criterion. Over-refinement rate is the "
            "main blocker metric: useful repair is not enough if later steps erase task-relevant signal."
        ),
    }


def _format_float(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = [
        "# Toy Refinement Stop Rules",
        "",
        "Deterministic ablation for stopping target-side latent refinement before over-repair.",
        "",
        "| Method | Accuracy | MSE | Reg MSE | Avg Steps | Compute | Bytes | Help | Harm | MSE Help | MSE Harm | Over-Refine | Confidence | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        formatted = {key: _format_float(value) for key, value in row.items() if key not in {"stop_reasons", "stop_histogram"}}
        lines.append(
            "| {method} | {task_accuracy} | {mse} | {regression_mse} | {average_steps} | "
            "{compute_proxy} | {bytes_proxy} | {help_rate} | {harm_rate} | {mse_help_rate} | "
            "{mse_harm_rate} | {over_refinement_rate} | {mean_confidence} | {confidence_ece} |".format(
                **formatted
            )
        )
    lines.extend(["", "## Stop Reasons", ""])
    for row in payload["rows"]:
        reasons = ", ".join(f"{key}={value}" for key, value in sorted(row["stop_reasons"].items()))
        histogram = ", ".join(f"{key}_steps={value}" for key, value in sorted(row["stop_histogram"].items()))
        lines.append(f"- `{row['method']}`: {reasons}; {histogram}")
    lines.extend(["", "## Interpretation", "", payload["interpretation"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--output-md", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=ToyRefinementStopRulesConfig.seed)
    parser.add_argument("--examples", type=int, default=ToyRefinementStopRulesConfig.examples)
    parser.add_argument("--dim", type=int, default=ToyRefinementStopRulesConfig.dim)
    parser.add_argument("--classes", type=int, default=ToyRefinementStopRulesConfig.classes)
    parser.add_argument("--styles", type=int, default=ToyRefinementStopRulesConfig.styles)
    parser.add_argument("--quant-bits", type=int, default=ToyRefinementStopRulesConfig.quant_bits)
    parser.add_argument("--bridge-noise", type=float, default=ToyRefinementStopRulesConfig.bridge_noise)
    parser.add_argument("--source-noise", type=float, default=ToyRefinementStopRulesConfig.source_noise)
    parser.add_argument("--refinement-rate", type=float, default=ToyRefinementStopRulesConfig.refinement_rate)
    parser.add_argument("--confidence-threshold", type=float, default=ToyRefinementStopRulesConfig.confidence_threshold)
    parser.add_argument("--score-drift-threshold", type=float, default=ToyRefinementStopRulesConfig.score_drift_threshold)
    parser.add_argument("--verifier-harm-margin", type=float, default=ToyRefinementStopRulesConfig.verifier_harm_margin)
    parser.add_argument("--max-steps", type=int, default=ToyRefinementStopRulesConfig.max_steps)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    config = ToyRefinementStopRulesConfig(
        seed=args.seed,
        examples=args.examples,
        dim=args.dim,
        classes=args.classes,
        styles=args.styles,
        quant_bits=args.quant_bits,
        bridge_noise=args.bridge_noise,
        source_noise=args.source_noise,
        refinement_rate=args.refinement_rate,
        confidence_threshold=args.confidence_threshold,
        score_drift_threshold=args.score_drift_threshold,
        verifier_harm_margin=args.verifier_harm_margin,
        max_steps=args.max_steps,
    )
    if config.max_steps != 4:
        raise ValueError("this ablation expects max_steps=4 so fixed 1/2/4 controls are comparable")
    payload = run_experiment(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.output_md)
    return payload


if __name__ == "__main__":
    main()
