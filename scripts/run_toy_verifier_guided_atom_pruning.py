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


METHODS: tuple[str, ...] = (
    "no_pruning",
    "scalar_score_pruning",
    "step_error_localized_pruning",
    "verifier_guided_frontier_pruning",
    "oracle_pruning",
)


@dataclass(frozen=True)
class ToyVerifierGuidedAtomPruningConfig:
    seed: int = 0
    train_examples: int = 160
    test_examples: int = 128
    dim: int = 16
    atoms: int = 18
    steps: int = 3
    helpful_atoms: int = 8
    harmful_atoms: int = 6
    keep_fraction: float = 0.5
    activation_temperature: float = 0.75
    scalar_noise: float = 0.35
    step_noise: float = 0.22
    verifier_noise: float = 0.10
    cost_jitter: float = 0.20
    step_bonus_scale: float = 0.45
    verifier_bonus_scale: float = 0.65


@dataclass(frozen=True)
class ToyExample:
    query: torch.Tensor
    activations: torch.Tensor
    utility: torch.Tensor
    atom_cost: torch.Tensor
    step_ids: torch.Tensor
    target_summary: torch.Tensor
    target_label: torch.Tensor


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _bytes_for_values(count: int, bits: float) -> float:
    return float(math.ceil(count * bits / 8.0))


def _compute_proxy(selected_cost: float, dim: int) -> float:
    return float(selected_cost * dim * 2.0)


def _bytes_proxy(selected_atoms: int, dim: int) -> float:
    return float(selected_atoms * dim * 4.0)


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _make_problem(config: ToyVerifierGuidedAtomPruningConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    atom_keys = _normalize_rows(torch.randn(config.atoms, config.dim, generator=gen))
    atom_values = torch.randn(config.atoms, config.dim, generator=gen)
    classifier = torch.randn(config.dim, 5, generator=gen) / math.sqrt(config.dim)

    helpful = min(max(int(config.helpful_atoms), 1), config.atoms)
    harmful = min(max(int(config.harmful_atoms), 0), config.atoms - helpful)
    neutral = config.atoms - helpful - harmful
    utility = torch.cat(
        [
            torch.linspace(1.35, 0.55, steps=helpful),
            -torch.linspace(0.95, 0.35, steps=harmful),
            0.12 * torch.randn(neutral, generator=gen),
        ],
        dim=0,
    )
    utility = utility[torch.randperm(config.atoms, generator=gen)]

    step_ids = torch.arange(config.atoms, dtype=torch.long) % max(config.steps, 1)
    step_ids = step_ids[torch.randperm(config.atoms, generator=gen)]
    atom_cost = 1.0 + 0.20 * step_ids.float() + config.cost_jitter * torch.rand(config.atoms, generator=gen)

    return {
        "atom_keys": atom_keys,
        "atom_values": atom_values,
        "classifier": classifier,
        "utility": utility.float(),
        "step_ids": step_ids,
        "atom_cost": atom_cost.float(),
    }


def _make_examples(
    config: ToyVerifierGuidedAtomPruningConfig,
    *,
    count: int,
    seed_offset: int,
    problem: dict[str, torch.Tensor],
) -> list[ToyExample]:
    gen = _make_generator(config.seed + seed_offset)
    atom_keys = problem["atom_keys"]
    atom_values = problem["atom_values"]
    utility = problem["utility"]
    step_ids = problem["step_ids"]
    atom_cost = problem["atom_cost"]

    examples: list[ToyExample] = []
    for _ in range(count):
        query = torch.randn(config.dim, generator=gen)
        key_scores = (atom_keys @ query) / math.sqrt(config.dim)
        activations = torch.softmax(key_scores / max(config.activation_temperature, 1e-8), dim=-1)
        activations = activations + 0.02 * torch.randn(config.atoms, generator=gen)
        activations = activations.clamp_min(0.0)
        activations = activations / activations.sum().clamp_min(1e-8)

        clean_contrib = activations * utility.clamp_min(0.0)
        target_summary = torch.einsum("a,ad->d", clean_contrib, atom_values)
        target_summary = target_summary / clean_contrib.sum().clamp_min(1e-8)
        target_summary = target_summary + 0.03 * torch.randn(config.dim, generator=gen)
        target_label = torch.argmax(target_summary @ problem["classifier"], dim=-1)

        examples.append(
            ToyExample(
                query=query.float(),
                activations=activations.float(),
                utility=utility.float(),
                atom_cost=atom_cost.float(),
                step_ids=step_ids.long(),
                target_summary=target_summary.float(),
                target_label=target_label.long(),
            )
        )
    return examples


def _build_dataset(
    config: ToyVerifierGuidedAtomPruningConfig,
) -> tuple[dict[str, torch.Tensor], list[ToyExample], list[ToyExample]]:
    problem = _make_problem(config)
    train = _make_examples(config, count=config.train_examples, seed_offset=10_000, problem=problem)
    test = _make_examples(config, count=config.test_examples, seed_offset=20_000, problem=problem)
    return problem, train, test


def _stack(field: str, examples: Sequence[ToyExample]) -> torch.Tensor:
    return torch.stack([getattr(example, field) for example in examples], dim=0)


def _step_bonus(examples: Sequence[ToyExample], *, scale: float) -> torch.Tensor:
    utility = _stack("utility", examples)
    step_ids = _stack("step_ids", examples)
    bonuses = []
    step_count = int(step_ids.max().item()) + 1 if step_ids.numel() else 1
    for idx in range(utility.shape[0]):
        step_mass = []
        for step in range(step_count):
            mask = step_ids[idx] == step
            if mask.any():
                step_mass.append(float((utility[idx][mask] * 1.0).sum().item()))
            else:
                step_mass.append(0.0)
        step_mass_t = torch.tensor(step_mass, dtype=utility.dtype)
        centered = step_mass_t - step_mass_t.mean()
        atom_bonus = centered[step_ids[idx]] * float(scale)
        bonuses.append(atom_bonus)
    return torch.stack(bonuses, dim=0)


def _predict_from_mask(example: ToyExample, keep_mask: torch.Tensor, atom_values: torch.Tensor) -> torch.Tensor:
    kept_acts = example.activations * keep_mask.float()
    denom = kept_acts.sum().clamp_min(1e-8)
    return torch.einsum("a,ad->d", kept_acts, atom_values) / denom


def _select_mask(
    scores: torch.Tensor,
    atom_cost: torch.Tensor,
    *,
    budget_cost: float,
    use_frontier: bool,
) -> torch.Tensor:
    if use_frontier:
        order = torch.argsort(scores / atom_cost.clamp_min(1e-8), descending=True)
    else:
        order = torch.argsort(scores, descending=True)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    spent = 0.0
    for idx in order.tolist():
        next_cost = spent + float(atom_cost[idx].item())
        if next_cost <= budget_cost or keep.sum().item() == 0:
            keep[idx] = True
            spent = next_cost
        if spent >= budget_cost:
            break
    if not bool(keep.any()):
        keep[order[0]] = True
    return keep


def _atom_recall(selected: torch.Tensor, oracle: torch.Tensor) -> float:
    if oracle.sum().item() == 0:
        return 1.0 if selected.sum().item() == 0 else 0.0
    return float((selected & oracle).float().sum().item() / max(float(oracle.float().sum().item()), 1.0))


def _evaluate_method(
    config: ToyVerifierGuidedAtomPruningConfig,
    problem: dict[str, torch.Tensor],
    examples: Sequence[ToyExample],
    method: str,
    *,
    step_bonus: torch.Tensor,
) -> list[dict[str, Any]]:
    atom_values = problem["atom_values"]
    rows: list[dict[str, Any]] = []
    total_cost = float(problem["atom_cost"].sum().item())
    budget_cost = float(config.keep_fraction) * total_cost

    for index, example in enumerate(examples):
        true_contrib = example.activations * example.utility
        clean_contrib = example.activations * example.utility.clamp_min(0.0)
        oracle_scores = clean_contrib.clone()
        scalar_noise = torch.randn(true_contrib.shape, generator=_make_generator(config.seed + 31_000 + index), dtype=true_contrib.dtype)
        step_noise = torch.randn(true_contrib.shape, generator=_make_generator(config.seed + 32_000 + index), dtype=true_contrib.dtype)
        verifier_noise = torch.randn(true_contrib.shape, generator=_make_generator(config.seed + 33_000 + index), dtype=true_contrib.dtype)
        scalar_scores = true_contrib + config.scalar_noise * scalar_noise
        step_scores = true_contrib + config.step_bonus_scale * step_bonus[index] + config.step_noise * step_noise
        verifier_scores = (
            true_contrib
            + 0.18 * config.verifier_bonus_scale * step_bonus[index]
            + 0.5 * config.verifier_noise * verifier_noise
        )

        if method == "no_pruning":
            keep_mask = torch.ones(config.atoms, dtype=torch.bool)
            selector_scores = None
        elif method == "scalar_score_pruning":
            keep_mask = _select_mask(scalar_scores, example.atom_cost, budget_cost=budget_cost, use_frontier=False)
            selector_scores = scalar_scores
        elif method == "step_error_localized_pruning":
            keep_mask = _select_mask(step_scores, example.atom_cost, budget_cost=budget_cost, use_frontier=False)
            selector_scores = step_scores
        elif method == "verifier_guided_frontier_pruning":
            keep_mask = _select_mask(verifier_scores, example.atom_cost, budget_cost=budget_cost, use_frontier=True)
            selector_scores = verifier_scores
        elif method == "oracle_pruning":
            keep_mask = _select_mask(oracle_scores, example.atom_cost, budget_cost=budget_cost, use_frontier=True)
            selector_scores = oracle_scores
        else:
            raise ValueError(f"Unknown method: {method}")

        predicted_summary = _predict_from_mask(example, keep_mask, atom_values)
        mse = float(F.mse_loss(predicted_summary, example.target_summary).item())
        pred_label = int(torch.argmax(predicted_summary @ problem["classifier"], dim=-1).item())
        accuracy = 1.0 if pred_label == int(example.target_label.item()) else 0.0

        helpful_mask = example.utility > 0
        harmful_mask = example.utility < 0
        pruned_mask = ~keep_mask
        oracle_keep_mask = _select_mask(oracle_scores, example.atom_cost, budget_cost=budget_cost, use_frontier=True)

        missed_help = float((pruned_mask & helpful_mask).float().sum().item() / max(helpful_mask.float().sum().item(), 1.0))
        false_prune = float((pruned_mask & helpful_mask).float().sum().item() / max(pruned_mask.float().sum().item(), 1.0))
        pruned_harm = float((pruned_mask & harmful_mask).float().sum().item() / max(harmful_mask.float().sum().item(), 1.0))
        prune_rate = float(pruned_mask.float().mean().item())
        selected_cost = float(example.atom_cost[keep_mask].sum().item())
        selected_atoms = int(keep_mask.sum().item())
        atom_recovery = _atom_recall(keep_mask, oracle_keep_mask)

        row = {
            "method": method,
            "example_index": index,
            "accuracy": accuracy,
            "mse": mse,
            "prune_rate": prune_rate,
            "kept_atoms": selected_atoms,
            "kept_cost": selected_cost,
            "missed_help_rate": missed_help,
            "false_prune_rate": false_prune,
            "pruned_harm_rate": pruned_harm,
            "atom_recovery": atom_recovery,
            "bytes_proxy": _bytes_proxy(selected_atoms, config.dim),
            "compute_proxy": _compute_proxy(selected_cost, config.dim),
            "selected_score_mean": float(selector_scores[keep_mask].mean().item()) if selector_scores is not None else None,
            "selected_score_margin": float(
                (
                    torch.sort(selector_scores, descending=True).values[:2]
                    if selector_scores is not None and selector_scores.numel() > 1
                    else torch.tensor([0.0], dtype=true_contrib.dtype)
                )[0].item()
                - (
                    torch.sort(selector_scores, descending=True).values[:2]
                    if selector_scores is not None and selector_scores.numel() > 1
                    else torch.tensor([0.0], dtype=true_contrib.dtype)
                )[-1].item()
            ),
        }
        rows.append(row)
    return rows


def run_experiment(config: ToyVerifierGuidedAtomPruningConfig) -> list[dict[str, Any]]:
    problem, train, test = _build_dataset(config)
    del train
    step_bonus = _step_bonus(test, scale=config.step_bonus_scale)

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows.extend(_evaluate_method(config, problem, test, method, step_bonus=step_bonus))

    grouped: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in rows:
        grouped[row["method"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    raw = grouped["no_pruning"]
    raw_accuracy = sum(row["accuracy"] for row in raw) / max(len(raw), 1)
    raw_mse = sum(row["mse"] for row in raw) / max(len(raw), 1)
    for method in METHODS:
        subset = grouped[method]
        accuracy = sum(row["accuracy"] for row in subset) / max(len(subset), 1)
        mse = sum(row["mse"] for row in subset) / max(len(subset), 1)
        prune_rate = sum(row["prune_rate"] for row in subset) / max(len(subset), 1)
        missed_help_rate = sum(row["missed_help_rate"] for row in subset) / max(len(subset), 1)
        false_prune_rate = sum(row["false_prune_rate"] for row in subset) / max(len(subset), 1)
        atom_recovery = sum(row["atom_recovery"] for row in subset) / max(len(subset), 1)
        bytes_proxy = sum(row["bytes_proxy"] for row in subset) / max(len(subset), 1)
        compute_proxy = sum(row["compute_proxy"] for row in subset) / max(len(subset), 1)
        help_vs_raw = max(0.0, accuracy - raw_accuracy)
        harm_vs_raw = max(0.0, raw_accuracy - accuracy)
        summary_rows.append(
            {
                "method": method,
                "accuracy": float(accuracy),
                "mse": float(mse),
                "accuracy_delta_vs_no_pruning": float(accuracy - raw_accuracy),
                "mse_delta_vs_no_pruning": float(mse - raw_mse),
                "prune_rate": float(prune_rate),
                "missed_help_rate": float(missed_help_rate),
                "false_prune_rate": float(false_prune_rate),
                "atom_recovery": float(atom_recovery),
                "bytes_proxy": float(bytes_proxy),
                "compute_proxy": float(compute_proxy),
                "help_vs_no_pruning": float(help_vs_raw),
                "harm_vs_no_pruning": float(harm_vs_raw),
            }
        )
    return summary_rows


def write_markdown_summary(config: ToyVerifierGuidedAtomPruningConfig, rows: list[dict[str, Any]], path: pathlib.Path) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    lines = [
        "# Toy Verifier-Guided Atom Pruning",
        "",
        f"- Seed: `{config.seed}`",
        f"- Train examples: `{config.train_examples}`",
        f"- Test examples: `{config.test_examples}`",
        f"- Atoms: `{config.atoms}`",
        f"- Steps: `{config.steps}`",
        f"- Keep fraction: `{config.keep_fraction}`",
        "",
        "| Method | Accuracy | MSE | Prune rate | Missed help | False prune | Atom recovery | Bytes proxy | Compute proxy | Help vs no-pruning | Harm vs no-pruning |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {mse} | {prune_rate} | {missed_help_rate} | {false_prune_rate} | {atom_recovery} | {bytes_proxy} | {compute_proxy} | {help_vs_no_pruning} | {harm_vs_no_pruning} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                mse=fmt(row["mse"]),
                prune_rate=fmt(row["prune_rate"]),
                missed_help_rate=fmt(row["missed_help_rate"]),
                false_prune_rate=fmt(row["false_prune_rate"]),
                atom_recovery=fmt(row["atom_recovery"]),
                bytes_proxy=fmt(row["bytes_proxy"]),
                compute_proxy=fmt(row["compute_proxy"]),
                help_vs_no_pruning=fmt(row["help_vs_no_pruning"]),
                harm_vs_no_pruning=fmt(row["harm_vs_no_pruning"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy verifier-guided atom pruning ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=160)
    parser.add_argument("--test-examples", type=int, default=128)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--atoms", type=int, default=18)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--helpful-atoms", type=int, default=8)
    parser.add_argument("--harmful-atoms", type=int, default=6)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--activation-temperature", type=float, default=0.75)
    parser.add_argument("--scalar-noise", type=float, default=0.35)
    parser.add_argument("--step-noise", type=float, default=0.22)
    parser.add_argument("--verifier-noise", type=float, default=0.10)
    parser.add_argument("--cost-jitter", type=float, default=0.20)
    parser.add_argument("--step-bonus-scale", type=float, default=0.45)
    parser.add_argument("--verifier-bonus-scale", type=float, default=0.65)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyVerifierGuidedAtomPruningConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        atoms=args.atoms,
        steps=args.steps,
        helpful_atoms=args.helpful_atoms,
        harmful_atoms=args.harmful_atoms,
        keep_fraction=args.keep_fraction,
        activation_temperature=args.activation_temperature,
        scalar_noise=args.scalar_noise,
        step_noise=args.step_noise,
        verifier_noise=args.verifier_noise,
        cost_jitter=args.cost_jitter,
        step_bonus_scale=args.step_bonus_scale,
        verifier_bonus_scale=args.verifier_bonus_scale,
    )
    rows = run_experiment(config)
    payload = {"config": asdict(config), "methods": list(METHODS), "rows": rows}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md:
        write_markdown_summary(config, rows, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
