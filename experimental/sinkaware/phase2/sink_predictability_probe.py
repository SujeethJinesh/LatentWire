"""Probe whether SinkAware can be revived as an approximate low-rank prior.

The exact static-prior branch is already killed because sink logits remain
query-dependent. This script checks the weaker question: under what query
geometry could a tiny approximate model predict fixed-sink logits well enough
to justify a future approximate/fused kernel branch?
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experimental/sinkaware/phase2"


def _make_queries(case: str, n: int, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if case == "random":
        return torch.randn(n, dim, generator=generator)
    if case == "clustered":
        centers = torch.randn(6, dim, generator=generator)
        assignments = torch.randint(0, centers.shape[0], (n,), generator=generator)
        return centers[assignments] + 0.15 * torch.randn(n, dim, generator=generator)
    if case == "low_rank":
        basis = torch.randn(3, dim, generator=generator)
        coeff = torch.randn(n, basis.shape[0], generator=generator)
        return coeff @ basis + 0.05 * torch.randn(n, dim, generator=generator)
    raise ValueError(f"unknown case: {case}")


def _r2(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0, keepdim=True)) ** 2)
    return float(1.0 - ss_res / ss_tot.clamp_min(1e-12))


def _fit_static(train_logits: torch.Tensor, test_logits: torch.Tensor) -> float:
    pred = train_logits.mean(dim=0, keepdim=True).expand_as(test_logits)
    return _r2(test_logits, pred)


def _fit_rank_query_model(
    train_q: torch.Tensor,
    train_logits: torch.Tensor,
    test_q: torch.Tensor,
    test_logits: torch.Tensor,
    rank: int,
    ridge: float = 1e-3,
) -> float:
    q_mean = train_q.mean(dim=0, keepdim=True)
    centered_train = train_q - q_mean
    centered_test = test_q - q_mean
    _, _, vh = torch.linalg.svd(centered_train, full_matrices=False)
    basis = vh[:rank].T
    x_train = centered_train @ basis
    x_test = centered_test @ basis
    x_train = torch.cat([x_train, torch.ones(x_train.shape[0], 1)], dim=1)
    x_test = torch.cat([x_test, torch.ones(x_test.shape[0], 1)], dim=1)
    eye = torch.eye(x_train.shape[1])
    weights = torch.linalg.solve(x_train.T @ x_train + ridge * eye, x_train.T @ train_logits)
    return _r2(test_logits, x_test @ weights)


def _run_case(case: str, seed: int, n: int = 768, dim: int = 64, n_sink: int = 4) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed + 10_000)
    queries = _make_queries(case, n=n, dim=dim, seed=seed)
    sink_keys = torch.randn(n_sink, dim, generator=generator) / dim**0.5
    logits = queries @ sink_keys.T
    split = int(0.67 * n)
    train_q, test_q = queries[:split], queries[split:]
    train_logits, test_logits = logits[:split], logits[split:]
    row: dict[str, object] = {
        "case": case,
        "seed": seed,
        "static_r2": _fit_static(train_logits, test_logits),
    }
    for rank in (1, 2, 4, 8):
        row[f"rank{rank}_query_r2"] = _fit_rank_query_model(train_q, train_logits, test_q, test_logits, rank=rank)
    return row


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    metrics = ["static_r2", "rank1_query_r2", "rank2_query_r2", "rank4_query_r2", "rank8_query_r2"]
    summary: dict[str, object] = {}
    for case in sorted({str(row["case"]) for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        summary[case] = {
            metric: sum(float(row[metric]) for row in case_rows) / len(case_rows)
            for metric in metrics
        }
    return summary


def _status(summary: dict[str, object]) -> str:
    random_static = float(summary["random"]["static_r2"])
    low_rank_r4 = float(summary["low_rank"]["rank4_query_r2"])
    clustered_r8 = float(summary["clustered"]["rank8_query_r2"])
    if random_static < 0.05 and low_rank_r4 > 0.95 and clustered_r8 > 0.50:
        return "REVIVE only as approximate low-rank/clustered query prior; exact static prior remains killed."
    return "DO NOT REVIVE without real-query evidence; synthetic predictability is too weak."


def _write_markdown(result: dict[str, object]) -> None:
    lines = [
        "# SinkAware Approximate Revival Gate",
        "",
        f"Status: **{result['status']}**",
        "",
        "The exact static sink-prior branch remains dead: fixed sink keys still need query-dependent `QK_sink`.",
        "This gate only tests whether an approximate low-rank or clustered-query prior might be worth a later real-query probe.",
        "",
        "| Query case | Static R2 | Rank-1 query R2 | Rank-2 query R2 | Rank-4 query R2 | Rank-8 query R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case, metrics in result["summary"].items():
        lines.append(
            "| {case} | {static_r2:.3f} | {rank1_query_r2:.3f} | {rank2_query_r2:.3f} | {rank4_query_r2:.3f} | {rank8_query_r2:.3f} |".format(
                case=case, **metrics
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Static priors are not revived. They do not explain query-dependent sink logits.",
            "The only branch still alive is approximate: exploit a low-dimensional or clustered query manifold, or fuse exact `QK_sink` computation more cheaply without pretending it can be skipped.",
            "The next gate must use real Q/K tensors or attention telemetry; synthetic geometry alone is not enough for a reviewer pack.",
        ]
    )
    (OUT_DIR / "sink_predictability_probe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_run_case(case, seed) for case in ("random", "clustered", "low_rank") for seed in range(8)]
    summary = _aggregate(rows)
    result = {"rows": rows, "summary": summary, "status": _status(summary)}
    (OUT_DIR / "sink_predictability_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result)
    print(json.dumps({"status": result["status"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
