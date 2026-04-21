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


SCENARIOS: tuple[str, ...] = (
    "identity",
    "permutation",
    "orthogonal_rotation",
    "permutation_rotation",
    "nonlinear_noise",
)

METHODS: tuple[str, ...] = (
    "identity",
    "permutation_only",
    "orthogonal_procrustes",
    "permutation_plus_procrustes",
    "ridge_stitch",
)


@dataclass(frozen=True)
class ToySymmetryConfig:
    seed: int = 0
    train_examples: int = 64
    test_examples: int = 64
    dim: int = 16
    noise: float = 0.04
    ridge_lam: float = 1e-2
    nonlinear_strength: float = 0.22


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _scenario_offset(name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(name))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _make_source(count: int, dim: int, *, seed: int) -> torch.Tensor:
    gen = _make_generator(seed)
    means = torch.linspace(-1.25, 1.25, dim, dtype=torch.float32)
    scales = torch.linspace(0.65, 1.55, dim, dtype=torch.float32)
    x = torch.randn(count, dim, generator=gen, dtype=torch.float32) * scales
    x = x + means
    if dim > 1:
        shared = torch.randn(count, 1, generator=gen, dtype=torch.float32)
        ramp = torch.linspace(-0.35, 0.35, dim, dtype=torch.float32)
        x = x + 0.25 * shared * ramp
    return x


def _feature_signature(x: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            x.mean(dim=0),
            x.std(dim=0, unbiased=False),
            x.abs().mean(dim=0),
        ],
        dim=1,
    )


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


def _estimate_permutation(source_train: torch.Tensor, target_train: torch.Tensor) -> torch.Tensor:
    source_signature = _feature_signature(source_train)
    target_signature = _feature_signature(target_train)
    cost = torch.cdist(source_signature.double(), target_signature.double(), p=2)
    source_to_target = _hungarian(cost)
    target_order = torch.empty_like(source_to_target)
    target_order[source_to_target] = torch.arange(source_to_target.numel(), dtype=torch.long)
    return target_order


def _center(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = x.mean(dim=0)
    return x - mean, mean


def _cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=-1).mean().item())


def _pair_recall_at_1(predicted: torch.Tensor, target: torch.Tensor) -> float:
    pred = F.normalize(predicted, dim=-1)
    tgt = F.normalize(target, dim=-1)
    scores = pred @ tgt.T
    winners = scores.argmax(dim=-1)
    return float((winners == torch.arange(scores.shape[0], device=scores.device)).float().mean().item())


def _scenario_data(
    config: ToySymmetryConfig,
    scenario: str,
    *,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")

    count = config.train_examples if split == "train" else config.test_examples
    base_seed = config.seed + _scenario_offset(scenario) + (0 if split == "train" else 50_000)
    transform_seed = config.seed + 100_000 + _scenario_offset(scenario)

    source = _make_source(count, config.dim, seed=base_seed)
    gen = _make_generator(transform_seed)

    perm: torch.Tensor | None = None
    if scenario in {"permutation", "permutation_rotation"}:
        perm = torch.randperm(config.dim, generator=gen)

    if scenario == "identity":
        target = source.clone()
    elif scenario == "permutation":
        assert perm is not None
        target = source[:, perm]
    elif scenario == "orthogonal_rotation":
        rotation = _orthogonal_matrix(config.dim, gen)
        target = source @ rotation
    elif scenario == "permutation_rotation":
        assert perm is not None
        rotation = _orthogonal_matrix(config.dim, gen)
        target = source[:, perm] @ rotation
    elif scenario == "nonlinear_noise":
        rotation = _orthogonal_matrix(config.dim, gen)
        mixing = _orthogonal_matrix(config.dim, gen)
        linear = source @ rotation
        nonlinear = torch.tanh(linear)
        residual = torch.tanh((source.pow(2) @ mixing) / float(config.dim))
        target = nonlinear + config.nonlinear_strength * residual
    else:  # pragma: no cover - guarded above
        raise ValueError(f"Unhandled scenario: {scenario}")

    if config.noise > 0:
        target = target + config.noise * torch.randn(
            target.shape,
            generator=gen,
            dtype=target.dtype,
        )
    return source.float(), target.float(), perm


def _fit_identity(source_train: torch.Tensor, target_train: torch.Tensor) -> dict[str, Any]:
    _, source_mean = _center(source_train)
    _, target_mean = _center(target_train)
    return {
        "kind": "identity",
        "source_mean": source_mean,
        "target_mean": target_mean,
    }


def _fit_permutation(source_train: torch.Tensor, target_train: torch.Tensor) -> dict[str, Any]:
    perm_order = _estimate_permutation(source_train, target_train)
    _, source_mean = _center(source_train)
    _, target_mean = _center(target_train)
    return {
        "kind": "permutation",
        "perm_order": perm_order,
        "source_mean": source_mean,
        "target_mean": target_mean,
    }


def _fit_procrustes(source_train: torch.Tensor, target_train: torch.Tensor) -> dict[str, Any]:
    source_centered, source_mean = _center(source_train)
    target_centered, target_mean = _center(target_train)
    weight = orthogonal_procrustes(source_centered, target_centered)
    return {
        "kind": "linear",
        "weight": weight,
        "source_mean": source_mean,
        "target_mean": target_mean,
    }


def _fit_permutation_plus_procrustes(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
) -> dict[str, Any]:
    perm_order = _estimate_permutation(source_train, target_train)
    source_perm = source_train[:, perm_order]
    source_centered, source_mean = _center(source_perm)
    target_centered, target_mean = _center(target_train)
    weight = orthogonal_procrustes(source_centered, target_centered)
    return {
        "kind": "permutation_linear",
        "perm_order": perm_order,
        "weight": weight,
        "source_mean": source_mean,
        "target_mean": target_mean,
    }


def _fit_ridge_stitch(
    source_train: torch.Tensor,
    target_train: torch.Tensor,
    *,
    ridge_lam: float,
) -> dict[str, Any]:
    source_centered, source_mean = _center(source_train)
    target_centered, target_mean = _center(target_train)
    weight = ridge_projection(source_centered, target_centered, lam=ridge_lam)
    return {
        "kind": "linear",
        "weight": weight,
        "source_mean": source_mean,
        "target_mean": target_mean,
    }


def _predict(model: dict[str, Any], source: torch.Tensor) -> torch.Tensor:
    kind = model["kind"]
    source_mean = model["source_mean"]
    target_mean = model["target_mean"]

    if kind == "identity":
        return source - source_mean + target_mean
    if kind == "permutation":
        perm_order = model["perm_order"]
        return source[:, perm_order] - source_mean[perm_order] + target_mean
    if kind == "linear":
        return (source - source_mean) @ model["weight"] + target_mean
    if kind == "permutation_linear":
        perm_order = model["perm_order"]
        source_perm = source[:, perm_order]
        return (source_perm - source_mean) @ model["weight"] + target_mean
    raise ValueError(f"Unknown model kind: {kind}")


def _metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mse": float(F.mse_loss(predicted, target).item()),
        "cosine": _cosine_mean(predicted, target),
        "pair_recall_at_1": _pair_recall_at_1(predicted, target),
    }


def _permutation_metrics(
    estimated: torch.Tensor | None,
    true_perm: torch.Tensor | None,
) -> dict[str, float | bool | None]:
    if estimated is None or true_perm is None:
        return {
            "permutation_accuracy": None,
            "permutation_exact_match": None,
        }
    accuracy = float((estimated == true_perm).float().mean().item())
    return {
        "permutation_accuracy": accuracy,
        "permutation_exact_match": bool(torch.equal(estimated, true_perm)),
    }


def run_experiment(config: ToySymmetryConfig, scenarios: Sequence[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in tuple(scenarios) if scenarios is not None else SCENARIOS:
        source_train, target_train, true_perm = _scenario_data(config, scenario, split="train")
        source_test, target_test, _ = _scenario_data(config, scenario, split="test")

        models = {
            "identity": _fit_identity(source_train, target_train),
            "permutation_only": _fit_permutation(source_train, target_train),
            "orthogonal_procrustes": _fit_procrustes(source_train, target_train),
            "permutation_plus_procrustes": _fit_permutation_plus_procrustes(
                source_train,
                target_train,
            ),
            "ridge_stitch": _fit_ridge_stitch(
                source_train,
                target_train,
                ridge_lam=config.ridge_lam,
            ),
        }

        estimated_perm = models["permutation_only"].get("perm_order")
        perm_metrics = _permutation_metrics(estimated_perm, true_perm)
        for method in METHODS:
            model = models[method]
            train_pred = _predict(model, source_train)
            test_pred = _predict(model, source_test)
            row = {
                "scenario": scenario,
                "method": method,
                "seed": int(config.seed),
                "dim": int(config.dim),
                "train_examples": int(config.train_examples),
                "test_examples": int(config.test_examples),
                "noise": float(config.noise),
                "ridge_lam": float(config.ridge_lam),
                "nonlinear_strength": float(config.nonlinear_strength),
                "train_mse": float(F.mse_loss(train_pred, target_train).item()),
                "test_mse": float(F.mse_loss(test_pred, target_test).item()),
                "train_cosine": _cosine_mean(train_pred, target_train),
                "test_cosine": _cosine_mean(test_pred, target_test),
                "train_pair_recall_at_1": _pair_recall_at_1(train_pred, target_train),
                "test_pair_recall_at_1": _pair_recall_at_1(test_pred, target_test),
                "permutation_accuracy": perm_metrics["permutation_accuracy"] if method in {"permutation_only", "permutation_plus_procrustes"} else None,
                "permutation_exact_match": perm_metrics["permutation_exact_match"] if method in {"permutation_only", "permutation_plus_procrustes"} else None,
            }
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scenario in SCENARIOS:
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        if not scenario_rows:
            continue
        best_test_mse = min(scenario_rows, key=lambda row: row["test_mse"])
        best_test_recall = max(scenario_rows, key=lambda row: row["test_pair_recall_at_1"])
        summary[scenario] = {
            "best_test_mse_method": best_test_mse["method"],
            "best_test_mse": float(best_test_mse["test_mse"]),
            "best_test_recall_method": best_test_recall["method"],
            "best_test_recall_at_1": float(best_test_recall["test_pair_recall_at_1"]),
        }
    return summary


def write_markdown_summary(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    path: pathlib.Path,
) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return value
        return f"{float(value):.4f}"

    lines = [
        "# Toy Symmetry Bridge",
        "",
        f"- seed: `{rows[0]['seed'] if rows else 0}`",
        f"- dim: `{rows[0]['dim'] if rows else 0}`",
        f"- train examples: `{rows[0]['train_examples'] if rows else 0}`",
        f"- test examples: `{rows[0]['test_examples'] if rows else 0}`",
        "",
        "| Scenario | Method | Train MSE | Test MSE | Train Cos | Test Cos | Train R@1 | Test R@1 | Perm Acc | Perm Exact |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {method} | {train_mse} | {test_mse} | {train_cosine} | {test_cosine} | {train_pair_recall_at_1} | {test_pair_recall_at_1} | {permutation_accuracy} | {permutation_exact_match} |".format(
                scenario=row["scenario"],
                method=row["method"],
                train_mse=fmt(row["train_mse"]),
                test_mse=fmt(row["test_mse"]),
                train_cosine=fmt(row["train_cosine"]),
                test_cosine=fmt(row["test_cosine"]),
                train_pair_recall_at_1=fmt(row["train_pair_recall_at_1"]),
                test_pair_recall_at_1=fmt(row["test_pair_recall_at_1"]),
                permutation_accuracy=fmt(row["permutation_accuracy"]),
                permutation_exact_match=fmt(row["permutation_exact_match"]),
            )
        )
    lines.extend(["", "## Best by Scenario"])
    for scenario in SCENARIOS:
        if scenario not in summary:
            continue
        item = summary[scenario]
        lines.append(
            f"- {scenario}: best test MSE = `{item['best_test_mse_method']}` ({item['best_test_mse']:.4f}), "
            f"best test recall = `{item['best_test_recall_method']}` ({item['best_test_recall_at_1']:.4f})",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact toy ablation for symmetry/orientation blockers.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--output-md", help="Markdown summary path. Defaults to the JSON path with a .md suffix.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=64)
    parser.add_argument("--test-examples", type=int, default=64)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.04)
    parser.add_argument("--ridge-lam", type=float, default=1e-2)
    parser.add_argument("--nonlinear-strength", type=float, default=0.22)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=list(SCENARIOS),
        help="Scenarios to run.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = ToySymmetryConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        noise=args.noise,
        ridge_lam=args.ridge_lam,
        nonlinear_strength=args.nonlinear_strength,
    )
    rows = run_experiment(config, args.scenarios)
    summary = summarize_rows(rows)
    payload = {
        "config": asdict(config),
        "scenarios": list(args.scenarios),
        "methods": list(METHODS),
        "summary": summary,
        "rows": rows,
    }

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    markdown_path = pathlib.Path(args.output_md) if args.output_md else output.with_suffix(".md")
    write_markdown_summary(rows, summary, markdown_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
