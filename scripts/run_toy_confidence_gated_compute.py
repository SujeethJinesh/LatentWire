#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ToyConfidenceGatedComputeConfig:
    seed: int = 0
    train_examples: int = 384
    test_examples: int = 192
    dim: int = 24
    classes: int = 6
    pool_size: int = 6
    max_budget: int = 4
    probe_noise_floor: float = 0.22
    probe_noise_span: float = 1.05
    pool_noise_floor: float = 0.14
    pool_noise_span: float = 1.15
    tail_shape: float = 1.35
    gate_temperature: float = 0.85
    target_avg_budget: float = 2.0
    budget_penalty: float = 0.14
    subgroup_bins: int = 3


@dataclass(frozen=True)
class ExampleOutcome:
    difficulty: float
    probe_confidence: float
    probe_correct: float
    probe_prediction: int
    selected_prediction: int
    selected_confidence: float
    selected_correct: float
    oracle_prediction: int
    oracle_confidence: float
    oracle_correct: float
    budget_1_prediction: int
    budget_1_confidence: float
    budget_1_correct: float
    budget_2_prediction: int
    budget_2_confidence: float
    budget_2_correct: float
    budget_4_prediction: int
    budget_4_confidence: float
    budget_4_correct: float


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def _orthogonal_matrix(dim: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=generator))
    signs = torch.sign(torch.diag(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.view(1, -1)


def _softmax_confidence(logits: torch.Tensor) -> tuple[int, float]:
    probs = torch.softmax(logits, dim=-1)
    confidence, prediction = probs.max(dim=-1)
    return int(prediction.item()), float(confidence.item())


def _make_problem(config: ToyConfidenceGatedComputeConfig) -> dict[str, torch.Tensor]:
    gen = _make_generator(config.seed)
    dim = int(config.dim)
    return {
        "true_w": torch.randn(dim, config.classes, generator=gen) / math.sqrt(float(dim)),
        "probe_w": torch.randn(dim, config.classes, generator=gen) / math.sqrt(float(dim)),
        "difficulty_w": torch.randn(dim, 1, generator=gen) / math.sqrt(float(dim)),
        "route_basis": _orthogonal_matrix(dim, gen),
    }


def _sample_example(
    config: ToyConfidenceGatedComputeConfig,
    base: dict[str, torch.Tensor],
    *,
    split: str,
    index: int,
) -> ExampleOutcome:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")
    offset = 10_000 if split == "test" else 0
    gen = _make_generator(config.seed + offset + 97 * index + sum(ord(ch) for ch in split))

    x = torch.randn(config.dim, generator=gen)
    base_logits = x @ base["true_w"]
    true_prediction = int(base_logits.argmax().item())

    difficulty_raw = torch.sigmoid((x @ base["difficulty_w"]).squeeze(-1))
    difficulty = float((0.18 + 0.82 * difficulty_raw.pow(1.7)).item())

    probe_noise_scale = config.probe_noise_floor + config.probe_noise_span * difficulty
    probe_logits = base_logits + probe_noise_scale * torch.randn(config.classes, generator=gen)
    probe_prediction, probe_confidence = _softmax_confidence(probe_logits)
    probe_correct = 1.0 if probe_prediction == true_prediction else 0.0

    candidate_logits: list[torch.Tensor] = [probe_logits]
    for _ in range(max(config.pool_size - 1, 0)):
        u = max(float(torch.rand((), generator=gen).item()), 1e-3)
        tail = 1.0 / (u ** (1.0 / max(config.tail_shape, 1e-3)))
        pool_noise_scale = config.pool_noise_floor + config.pool_noise_span / float(tail)
        candidate_logits.append(base_logits + pool_noise_scale * torch.randn(config.classes, generator=gen))

    if len(candidate_logits) < 4:
        raise ValueError("pool_size must be at least 4 to support the fixed-budget controls")

    candidate_outcomes = []
    for logits in candidate_logits:
        pred, conf = _softmax_confidence(logits)
        candidate_outcomes.append((pred, conf, 1.0 if pred == true_prediction else 0.0))

    def pick(prefix: int) -> tuple[int, float, float]:
        subset = candidate_outcomes[:prefix]
        best = max(subset, key=lambda item: (item[1], item[2]))
        return best

    budget_1_pred, budget_1_conf, budget_1_correct = pick(1)
    budget_2_pred, budget_2_conf, budget_2_correct = pick(2)
    budget_4_pred, budget_4_conf, budget_4_correct = pick(4)
    correct_candidates = [outcome for outcome in candidate_outcomes if outcome[2] >= 0.5]
    oracle_pred, oracle_conf, oracle_correct = max(
        correct_candidates if correct_candidates else candidate_outcomes,
        key=lambda item: (item[1], item[2]),
    )

    return ExampleOutcome(
        difficulty=difficulty,
        probe_confidence=probe_confidence,
        probe_correct=probe_correct,
        probe_prediction=probe_prediction,
        selected_prediction=budget_1_pred,
        selected_confidence=budget_1_conf,
        selected_correct=budget_1_correct,
        oracle_prediction=oracle_pred,
        oracle_confidence=oracle_conf,
        oracle_correct=oracle_correct,
        budget_1_prediction=budget_1_pred,
        budget_1_confidence=budget_1_conf,
        budget_1_correct=budget_1_correct,
        budget_2_prediction=budget_2_pred,
        budget_2_confidence=budget_2_conf,
        budget_2_correct=budget_2_correct,
        budget_4_prediction=budget_4_pred,
        budget_4_confidence=budget_4_conf,
        budget_4_correct=budget_4_correct,
    )


def _make_split(
    config: ToyConfidenceGatedComputeConfig,
    base: dict[str, torch.Tensor],
    *,
    split: str,
    count: int,
) -> list[ExampleOutcome]:
    return [_sample_example(config, base, split=split, index=i) for i in range(count)]


def _ece(confidences: Sequence[float], labels: Sequence[float], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    conf = torch.tensor(confidences, dtype=torch.float32)
    lab = torch.tensor(labels, dtype=torch.float32)
    edges = torch.linspace(0.0, 1.0, bins + 1)
    total = conf.numel()
    ece = 0.0
    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        if i < bins - 1:
            mask = (conf >= left) & (conf < right)
        else:
            mask = (conf >= left) & (conf <= right)
        if not mask.any():
            continue
        bin_conf = conf[mask].mean()
        bin_acc = lab[mask].mean()
        ece += float(mask.float().mean().item()) * float((bin_conf - bin_acc).abs().item())
    return ece


def _brier(confidences: Sequence[float], labels: Sequence[float]) -> float:
    if not confidences:
        return 0.0
    conf = torch.tensor(confidences, dtype=torch.float32)
    lab = torch.tensor(labels, dtype=torch.float32)
    return float(F.mse_loss(conf, lab).item())


def _auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
    if not scores:
        return 0.5
    pos = [(s, y) for s, y in zip(scores, labels) if y >= 0.5]
    neg = [(s, y) for s, y in zip(scores, labels) if y < 0.5]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    ties = 0.0
    for s_pos, _ in pos:
        for s_neg, _ in neg:
            if s_pos > s_neg:
                wins += 1.0
            elif s_pos == s_neg:
                ties += 1.0
    total = float(len(pos) * len(neg))
    return float((wins + 0.5 * ties) / total)


def _confidence_gap(confidences: Sequence[float], labels: Sequence[float]) -> float:
    if not confidences:
        return 0.0
    conf = torch.tensor(confidences, dtype=torch.float32)
    lab = torch.tensor(labels, dtype=torch.float32)
    correct = conf[lab >= 0.5]
    incorrect = conf[lab < 0.5]
    if correct.numel() == 0 or incorrect.numel() == 0:
        return 0.0
    return float(correct.mean().item() - incorrect.mean().item())


def _policy_budget(probe_confidence: float, *, low: float, high: float) -> int:
    if probe_confidence >= high:
        return 1
    if probe_confidence >= low:
        return 2
    return 4


def _apply_gate_temperature(confidence: float, temperature: float) -> float:
    temperature = max(float(temperature), 1e-3)
    confidence = min(max(float(confidence), 1e-6), 1.0 - 1e-6)
    logit = math.log(confidence / (1.0 - confidence))
    return float(1.0 / (1.0 + math.exp(-logit / temperature)))


def _evaluate_budgets(
    examples: Sequence[ExampleOutcome],
    *,
    fixed_budget: int | None = None,
    threshold_low: float | None = None,
    threshold_high: float | None = None,
    random_budget_probs: dict[int, float] | None = None,
    seed: int = 0,
    gate_temperature: float = 1.0,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    probe_confidences: list[float] = []
    probe_corrects: list[float] = []
    selected_confidences: list[float] = []
    selected_corrects: list[float] = []
    budgets: list[int] = []
    selected_predictions: list[int] = []

    cumulative_probs: list[tuple[int, float]] = []
    if random_budget_probs is not None:
        running = 0.0
        for budget in (1, 2, 4):
            running += float(random_budget_probs.get(budget, 0.0))
            cumulative_probs.append((budget, running))

    for index, example in enumerate(examples):
        if fixed_budget is not None:
            budget = int(fixed_budget)
        elif threshold_low is not None and threshold_high is not None:
            gated_confidence = _apply_gate_temperature(example.probe_confidence, gate_temperature)
            budget = _policy_budget(gated_confidence, low=threshold_low, high=threshold_high)
        elif random_budget_probs is not None:
            rng = random.Random(seed + 17 * index + 101)
            u = rng.random()
            budget = 4
            for candidate_budget, cutoff in cumulative_probs:
                if u <= cutoff:
                    budget = candidate_budget
                    break
        else:
            raise ValueError("Either thresholds or random_budget_probs must be provided")

        if budget == 1:
            pred = example.budget_1_prediction
            conf = example.budget_1_confidence
            correct = example.budget_1_correct
        elif budget == 2:
            pred = example.budget_2_prediction
            conf = example.budget_2_confidence
            correct = example.budget_2_correct
        elif budget == 4:
            pred = example.budget_4_prediction
            conf = example.budget_4_confidence
            correct = example.budget_4_correct
        else:  # pragma: no cover - budget choices are discrete.
            raise ValueError(budget)

        rows.append(
            {
                "difficulty": example.difficulty,
                "probe_confidence": example.probe_confidence,
                "probe_correct": example.probe_correct,
                "selected_prediction": pred,
                "selected_confidence": conf,
                "selected_correct": correct,
                "budget": budget,
                "oracle_correct": example.oracle_correct,
                "oracle_confidence": example.oracle_confidence,
            }
        )
        probe_confidences.append(example.probe_confidence)
        probe_corrects.append(example.probe_correct)
        selected_confidences.append(conf)
        selected_corrects.append(correct)
        budgets.append(budget)
        selected_predictions.append(pred)

    accuracy = float(sum(selected_corrects) / max(len(selected_corrects), 1))
    avg_budget = float(sum(budgets) / max(len(budgets), 1))
    oracle_accuracy = float(sum(example.oracle_correct for example in examples) / max(len(examples), 1))
    probe_accuracy = float(sum(probe_corrects) / max(len(probe_corrects), 1))
    budget_fraction = avg_budget / 4.0
    budget_histogram = {str(budget): budgets.count(budget) / max(len(budgets), 1) for budget in (1, 2, 4)}

    return {
        "rows": rows,
        "accuracy": accuracy,
        "avg_budget": avg_budget,
        "compute_fraction": budget_fraction,
        "oracle_accuracy": oracle_accuracy,
        "oracle_gap": float(oracle_accuracy - accuracy),
        "probe_accuracy": probe_accuracy,
        "probe_ece": _ece(probe_confidences, probe_corrects),
        "probe_brier": _brier(probe_confidences, probe_corrects),
        "probe_auroc": _auroc(probe_confidences, probe_corrects),
        "probe_confidence_gap": _confidence_gap(probe_confidences, probe_corrects),
        "selected_ece": _ece(selected_confidences, selected_corrects),
        "selected_brier": _brier(selected_confidences, selected_corrects),
        "selected_auroc": _auroc(selected_confidences, selected_corrects),
        "selected_confidence_gap": _confidence_gap(selected_confidences, selected_corrects),
        "budget_histogram": budget_histogram,
        "selected_predictions": selected_predictions,
    }


def _calibrate_thresholds(
    train_examples: Sequence[ExampleOutcome],
    config: ToyConfidenceGatedComputeConfig,
) -> dict[str, Any]:
    confidences = torch.tensor([example.probe_confidence for example in train_examples], dtype=torch.float32)
    quantiles = torch.unique(torch.quantile(confidences, torch.linspace(0.1, 0.9, 9)))
    if quantiles.numel() < 2:
        quantiles = torch.tensor([0.2, 0.8], dtype=torch.float32)

    best: dict[str, Any] | None = None
    for low in quantiles.tolist():
        for high in quantiles.tolist():
            if low >= high:
                continue
            stats = _evaluate_budgets(
                train_examples,
                threshold_low=float(low),
                threshold_high=float(high),
                gate_temperature=config.gate_temperature,
            )
            score = stats["accuracy"] - config.budget_penalty * abs(stats["avg_budget"] - config.target_avg_budget) / max(
                config.target_avg_budget, 1e-8
            )
            candidate = {
                "low_threshold": float(low),
                "high_threshold": float(high),
                "train_accuracy": stats["accuracy"],
                "train_avg_budget": stats["avg_budget"],
                "train_compute_fraction": stats["compute_fraction"],
                "train_score": float(score),
                "train_budget_histogram": stats["budget_histogram"],
                "train_examples": len(train_examples),
            }
            if best is None:
                best = candidate
                continue
            better_score = candidate["train_score"] > best["train_score"]
            same_score = math.isclose(candidate["train_score"], best["train_score"], rel_tol=1e-9, abs_tol=1e-9)
            closer_budget = abs(candidate["train_avg_budget"] - config.target_avg_budget) < abs(
                best["train_avg_budget"] - config.target_avg_budget
            )
            higher_accuracy = candidate["train_accuracy"] > best["train_accuracy"]
            if better_score or (same_score and (higher_accuracy or (math.isclose(candidate["train_accuracy"], best["train_accuracy"], rel_tol=1e-9, abs_tol=1e-9) and closer_budget))):
                best = candidate

    assert best is not None
    return best


def _match_random_budget_probs(train_stats: dict[str, Any]) -> dict[int, float]:
    histogram = train_stats["train_budget_histogram"]
    return {1: float(histogram.get("1", 0.0)), 2: float(histogram.get("2", 0.0)), 4: float(histogram.get("4", 0.0))}


def _subgroup_rows(
    examples: Sequence[ExampleOutcome],
    eval_rows: Sequence[dict[str, Any]],
    *,
    bins: int,
) -> dict[str, list[dict[str, Any]]]:
    def make_rows(values: Sequence[float], label: str) -> list[dict[str, Any]]:
        tensor = torch.tensor(values, dtype=torch.float32)
        boundaries = torch.quantile(tensor, torch.linspace(0.0, 1.0, bins + 1))
        rows: list[dict[str, Any]] = []
        for i in range(bins):
            left = float(boundaries[i].item())
            right = float(boundaries[i + 1].item())
            if i < bins - 1:
                indices = [j for j, value in enumerate(values) if value >= left and value < right]
            else:
                indices = [j for j, value in enumerate(values) if value >= left and value <= right]
            if not indices:
                continue
            subset_examples = [examples[j] for j in indices]
            subset_rows = [eval_rows[j] for j in indices]
            rows.append(
                {
                    "group": f"{label}_{i}",
                    "count": len(indices),
                    "accuracy": float(sum(row["selected_correct"] for row in subset_rows) / len(indices)),
                    "avg_budget": float(sum(row["budget"] for row in subset_rows) / len(indices)),
                    "oracle_gap": float(
                        sum(row["oracle_correct"] for row in subset_rows) / len(indices)
                        - sum(row["selected_correct"] for row in subset_rows) / len(indices)
                    ),
                    "probe_ece": _ece([ex.probe_confidence for ex in subset_examples], [ex.probe_correct for ex in subset_examples]),
                    "selected_ece": _ece(
                        [row["selected_confidence"] for row in subset_rows],
                        [row["selected_correct"] for row in subset_rows],
                    ),
                }
            )
        return rows

    return {
        "difficulty": make_rows([example.difficulty for example in examples], "difficulty"),
        "probe_confidence": make_rows([example.probe_confidence for example in examples], "probe_confidence"),
    }


def run_experiment(config: ToyConfidenceGatedComputeConfig) -> dict[str, Any]:
    base = _make_problem(config)
    train_examples = _make_split(config, base, split="train", count=config.train_examples)
    test_examples = _make_split(config, base, split="test", count=config.test_examples)

    calibration = _calibrate_thresholds(train_examples, config)
    gated = _evaluate_budgets(
        test_examples,
        threshold_low=calibration["low_threshold"],
        threshold_high=calibration["high_threshold"],
        gate_temperature=config.gate_temperature,
    )
    random_matched = _evaluate_budgets(
        test_examples,
        random_budget_probs=_match_random_budget_probs(calibration),
        seed=config.seed + 404,
    )
    fixed_1 = _evaluate_budgets(test_examples, fixed_budget=1)
    fixed_2 = _evaluate_budgets(test_examples, fixed_budget=2)
    fixed_4 = _evaluate_budgets(test_examples, fixed_budget=4)

    methods = {
        "fixed_budget_1": fixed_1,
        "fixed_budget_2": fixed_2,
        "fixed_budget_4": fixed_4,
        "random_budget_matched": random_matched,
        "confidence_gated": gated,
    }

    rows: list[dict[str, Any]] = []
    for method_name, stats in methods.items():
        rows.append(
            {
                "method": method_name,
                "accuracy": stats["accuracy"],
                "avg_budget": stats["avg_budget"],
                "compute_fraction": stats["compute_fraction"],
                "oracle_accuracy": stats["oracle_accuracy"],
                "oracle_gap": stats["oracle_gap"],
                "probe_accuracy": stats["probe_accuracy"],
                "probe_ece": stats["probe_ece"],
                "probe_brier": stats["probe_brier"],
                "probe_auroc": stats["probe_auroc"],
                "probe_confidence_gap": stats["probe_confidence_gap"],
                "selected_ece": stats["selected_ece"],
                "selected_brier": stats["selected_brier"],
                "selected_auroc": stats["selected_auroc"],
                "selected_confidence_gap": stats["selected_confidence_gap"],
                "budget_histogram": stats["budget_histogram"],
            }
        )

    payload = {
        "config": asdict(config),
        "calibration": calibration,
        "rows": rows,
        "subgroups": {
            "confidence_gated": _subgroup_rows(test_examples, gated["rows"], bins=config.subgroup_bins),
        },
    }
    return payload


def write_markdown_summary(payload: dict[str, Any], path: pathlib.Path) -> None:
    rows = payload["rows"]
    calibration = payload["calibration"]
    subgroups = payload["subgroups"]["confidence_gated"]

    def fmt(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return "-"
        return f"{float(value):.4f}"

    lines = [
        "# Toy Confidence-Gated Compute",
        "",
        f"- Seed: `{payload['config']['seed']}`",
        f"- Train examples: `{payload['config']['train_examples']}`",
        f"- Test examples: `{payload['config']['test_examples']}`",
        f"- Calibrated thresholds: low `{fmt(calibration['low_threshold'])}`, high `{fmt(calibration['high_threshold'])}`",
        f"- Train accuracy: `{fmt(calibration['train_accuracy'])}`",
        f"- Train avg budget: `{fmt(calibration['train_avg_budget'])}`",
        "",
        "| Method | Accuracy | Avg budget | Compute fraction | Oracle gap | Probe ECE | Selected ECE | Probe AUROC | Selected AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {accuracy} | {avg_budget} | {compute_fraction} | {oracle_gap} | {probe_ece} | {selected_ece} | {probe_auroc} | {selected_auroc} |".format(
                method=row["method"],
                accuracy=fmt(row["accuracy"]),
                avg_budget=fmt(row["avg_budget"]),
                compute_fraction=fmt(row["compute_fraction"]),
                oracle_gap=fmt(row["oracle_gap"]),
                probe_ece=fmt(row["probe_ece"]),
                selected_ece=fmt(row["selected_ece"]),
                probe_auroc=fmt(row["probe_auroc"]),
                selected_auroc=fmt(row["selected_auroc"]),
            )
        )
    lines.extend(
        [
            "",
            "## Confidence-Gated Subgroups",
            "",
            "| Group | Count | Accuracy | Avg budget | Oracle gap | Probe ECE | Selected ECE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in subgroups["difficulty"]:
        lines.append(
            "| {group} | {count} | {accuracy} | {avg_budget} | {oracle_gap} | {probe_ece} | {selected_ece} |".format(
                group=row["group"],
                count=row["count"],
                accuracy=fmt(row["accuracy"]),
                avg_budget=fmt(row["avg_budget"]),
                oracle_gap=fmt(row["oracle_gap"]),
                probe_ece=fmt(row["probe_ece"]),
                selected_ece=fmt(row["selected_ece"]),
            )
        )
    lines.extend(
        [
            "",
            "| Probe group | Count | Accuracy | Avg budget | Oracle gap | Probe ECE | Selected ECE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in subgroups["probe_confidence"]:
        lines.append(
            "| {group} | {count} | {accuracy} | {avg_budget} | {oracle_gap} | {probe_ece} | {selected_ece} |".format(
                group=row["group"],
                count=row["count"],
                accuracy=fmt(row["accuracy"]),
                avg_budget=fmt(row["avg_budget"]),
                oracle_gap=fmt(row["oracle_gap"]),
                probe_ece=fmt(row["probe_ece"]),
                selected_ece=fmt(row["selected_ece"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toy confidence-gated compute ablation.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-examples", type=int, default=384)
    parser.add_argument("--test-examples", type=int, default=192)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--pool-size", type=int, default=6)
    parser.add_argument("--max-budget", type=int, default=4)
    parser.add_argument("--probe-noise-floor", type=float, default=0.22)
    parser.add_argument("--probe-noise-span", type=float, default=1.05)
    parser.add_argument("--pool-noise-floor", type=float, default=0.14)
    parser.add_argument("--pool-noise-span", type=float, default=1.15)
    parser.add_argument("--tail-shape", type=float, default=1.35)
    parser.add_argument("--gate-temperature", type=float, default=0.85)
    parser.add_argument("--target-avg-budget", type=float, default=2.0)
    parser.add_argument("--budget-penalty", type=float, default=0.14)
    parser.add_argument("--subgroup-bins", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    config = ToyConfidenceGatedComputeConfig(
        seed=args.seed,
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        dim=args.dim,
        classes=args.classes,
        pool_size=args.pool_size,
        max_budget=args.max_budget,
        probe_noise_floor=args.probe_noise_floor,
        probe_noise_span=args.probe_noise_span,
        pool_noise_floor=args.pool_noise_floor,
        pool_noise_span=args.pool_noise_span,
        tail_shape=args.tail_shape,
        gate_temperature=args.gate_temperature,
        target_avg_budget=args.target_avg_budget,
        budget_penalty=args.budget_penalty,
        subgroup_bins=args.subgroup_bins,
    )
    payload = run_experiment(config)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        write_markdown_summary(payload, pathlib.Path(args.output_md))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
