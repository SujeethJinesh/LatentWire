#!/usr/bin/env python3
"""Build a compact evidence ladder from local ablation telemetry.

The output is not a paper result table. It is an anti-loop artifact that keeps
positive toy clues, failed controls, and promotion gates visible before we
spend time scaling a component into the real route-pool harness.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ACCURACY_KEYS = ("task_accuracy", "test_accuracy", "accuracy")
MSE_KEYS = ("mse", "test_mse")
BYTES_KEYS = ("bytes_proxy", "parameter_proxy")


@dataclass(frozen=True)
class EvidenceSpec:
    lane: str
    method: str
    artifact: str
    evidence_level: str
    baseline_method: str | None
    promotion_gate: str


@dataclass(frozen=True)
class EvidenceRow:
    spec: EvidenceSpec
    status: str
    accuracy: float | None = None
    delta_vs_baseline: float | None = None
    mse: float | None = None
    bytes_proxy: float | None = None
    atom_recovery: float | None = None
    route_accuracy: float | None = None
    route_stability: float | None = None
    over_refinement: float | None = None
    bit_histogram: str = ""
    help_rate: float | None = None
    harm_rate: float | None = None
    note: str = ""


DEFAULT_SPECS: tuple[EvidenceSpec, ...] = (
    EvidenceSpec(
        lane="Hub/shared dictionary",
        method="hub_shared_dictionary",
        artifact="results/query_pool_toy_20260421/hub_dictionary_bridge_20260421.json",
        evidence_level="toy positive",
        baseline_method="pairwise_bridges",
        promotion_gate="Beat pairwise on held-out route pools with fewer adapters or bytes.",
    ),
    EvidenceSpec(
        lane="Pairwise bridge control",
        method="pairwise_bridges",
        artifact="results/query_pool_toy_20260421/hub_dictionary_bridge_20260421.json",
        evidence_level="toy control",
        baseline_method="monolithic_bridge",
        promotion_gate="Keep only as O(n^2) scaling baseline.",
    ),
    EvidenceSpec(
        lane="Sticky feature routing",
        method="sticky_paraphrase_stable_routing",
        artifact="results/query_pool_toy_20260421/router_stability_regularization_20260421.json",
        evidence_level="toy positive",
        baseline_method="hard_feature_routing",
        promotion_gate="Improve perturbation stability without lowering route-pool accuracy.",
    ),
    EvidenceSpec(
        lane="Confidence-only routing",
        method="confidence_routing",
        artifact="results/query_pool_toy_20260421/router_stability_regularization_20260421.json",
        evidence_level="toy blocker",
        baseline_method="hard_feature_routing",
        promotion_gate="Do not rerun as sole router; only use as uncertainty feature.",
    ),
    EvidenceSpec(
        lane="Feature+atom stack",
        method="stacked_feature_atom",
        artifact="results/query_pool_toy_20260421/feature_atom_stack_bridge_20260421.json",
        evidence_level="toy positive interaction",
        baseline_method="raw_ridge",
        promotion_gate="Test interaction terms, not isolated feature-only or atom-only branches.",
    ),
    EvidenceSpec(
        lane="Mixed-bit frontier",
        method="quant_error_target_bpw_allocator",
        artifact="results/query_pool_toy_20260421/mixed_bit_route_atom_allocator_20260421.json",
        evidence_level="toy positive",
        baseline_method="uniform_3_bit",
        promotion_gate="Preserve accuracy at lower bpw than flat precision with help/harm logged.",
    ),
    EvidenceSpec(
        lane="Verifier stop rule",
        method="verifier_harm_stop",
        artifact="results/query_pool_toy_20260421/refinement_stop_rules_20260421.json",
        evidence_level="toy positive / safety",
        baseline_method="fixed_4_step",
        promotion_gate="Reduce over-refinement and harm versus fixed-depth repair.",
    ),
    EvidenceSpec(
        lane="Naive component stack",
        method="hub_sticky_frontier_verifier_stop",
        artifact="results/toy_hub_sticky_frontier_stack_20260421/toy_hub_sticky_frontier_stack.json",
        evidence_level="toy interaction blocker",
        baseline_method="raw_pairwise_bridge",
        promotion_gate="Do not stack hub, router, frontier, and stop policy until each interface is validated.",
    ),
    EvidenceSpec(
        lane="Stack oracle routing",
        method="oracle_router_control",
        artifact="results/toy_hub_sticky_frontier_stack_20260421/toy_hub_sticky_frontier_stack.json",
        evidence_level="toy oracle headroom",
        baseline_method="raw_pairwise_bridge",
        promotion_gate="Use as route-quality ceiling, not as a method row.",
    ),
    EvidenceSpec(
        lane="Fixed-depth repair blocker",
        method="fixed_4_step",
        artifact="results/query_pool_toy_20260421/refinement_stop_rules_20260421.json",
        evidence_level="toy blocker",
        baseline_method="fixed_1_step",
        promotion_gate="Do not promote without a stop policy.",
    ),
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _find_row(payload: dict[str, Any], method: str) -> dict[str, Any] | None:
    for row in _rows(payload):
        if row.get("method") == method:
            return row
    return None


def _first_number(row: dict[str, Any] | None, keys: Iterable[str]) -> float | None:
    if row is None:
        return None
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _bit_histogram(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    value = row.get("bit_allocation_histogram")
    if not isinstance(value, dict):
        return ""
    return ", ".join(f"{key}:{value[key]}" for key in sorted(value, key=lambda item: float(item)))


def summarize_spec(spec: EvidenceSpec, root: Path) -> EvidenceRow:
    artifact = root / spec.artifact
    payload = _read_json(artifact)
    if payload is None:
        return EvidenceRow(spec=spec, status="missing", note="artifact missing")
    row = _find_row(payload, spec.method)
    if row is None:
        return EvidenceRow(spec=spec, status="missing", note="method missing")

    baseline = _find_row(payload, spec.baseline_method) if spec.baseline_method else None
    accuracy = _first_number(row, ACCURACY_KEYS)
    baseline_accuracy = _first_number(baseline, ACCURACY_KEYS)
    delta = None
    if accuracy is not None and baseline_accuracy is not None:
        delta = accuracy - baseline_accuracy

    return EvidenceRow(
        spec=spec,
        status="present",
        accuracy=accuracy,
        delta_vs_baseline=delta,
        mse=_first_number(row, MSE_KEYS),
        bytes_proxy=_first_number(row, BYTES_KEYS),
        atom_recovery=_first_number(row, ("atom_recovery", "shared_feature_recovery")),
        route_accuracy=_first_number(row, ("route_accuracy",)),
        route_stability=_first_number(row, ("perturbation_stability", "stability")),
        over_refinement=_first_number(row, ("over_refinement_rate",)),
        bit_histogram=_bit_histogram(row),
        help_rate=_first_number(
            row,
            ("help_rate", "help_vs_raw", "help_vs_monolithic", "help_vs_uniform_3_bit", "help_vs_raw_pairwise"),
        ),
        harm_rate=_first_number(
            row,
            ("harm_rate", "harm_vs_raw", "harm_vs_monolithic", "harm_vs_uniform_3_bit", "harm_vs_raw_pairwise"),
        ),
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def render_markdown(rows: Sequence[EvidenceRow]) -> str:
    lines = [
        "# Ablation Evidence Ladder",
        "",
        "Date: 2026-04-21",
        "",
        "This table summarizes local telemetry for stack decisions. It separates toy-positive components, controls, and blockers so we do not rerun saturated ideas without changing the hypothesis.",
        "",
        "| Lane | Method | Level | Status | Accuracy | Delta | MSE | Bytes | Atom recovery | Route acc | Stability | Over-refine | Bits | Help | Harm | Promotion gate |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.spec.lane,
                    f"`{row.spec.method}`",
                    row.spec.evidence_level,
                    row.status if not row.note else f"{row.status}: {row.note}",
                    _fmt(row.accuracy),
                    _fmt(row.delta_vs_baseline),
                    _fmt(row.mse),
                    _fmt(row.bytes_proxy),
                    _fmt(row.atom_recovery),
                    _fmt(row.route_accuracy),
                    _fmt(row.route_stability),
                    _fmt(row.over_refinement),
                    row.bit_histogram or "-",
                    _fmt(row.help_rate),
                    _fmt(row.harm_rate),
                    row.spec.promotion_gate,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Read",
            "",
            "- Promote hub dictionaries, sticky/feature routing, mixed-bit frontiers, and verifier stop rules only as an interaction stack with matched controls.",
            "- Treat confidence-only routing and fixed-depth repair as blockers, not baselines to keep rerunning.",
            "- Any real-route-pool promotion should carry the same telemetry columns: atom recovery, route stability, bit histogram, stop reason, help/harm, bytes, and latency.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_rows(root: Path, specs: Sequence[EvidenceSpec] = DEFAULT_SPECS) -> list[EvidenceRow]:
    return [summarize_spec(spec, root) for spec in specs]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/ablation_evidence_ladder_20260421.md"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    markdown = render_markdown(build_rows(args.repo_root))
    output = args.repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
