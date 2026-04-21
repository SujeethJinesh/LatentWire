from __future__ import annotations

import json
from pathlib import Path

from scripts import build_ablation_evidence_table as table


def _write_payload(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}, sort_keys=True), encoding="utf-8")


def test_summarize_spec_extracts_metrics_and_delta(tmp_path: Path) -> None:
    artifact = tmp_path / "results/toy.json"
    _write_payload(
        artifact,
        [
            {"method": "baseline", "task_accuracy": 0.25, "mse": 1.0},
            {
                "method": "candidate",
                "task_accuracy": 0.75,
                "mse": 0.2,
                "bytes_proxy": 16,
                "atom_recovery": 0.9,
                "route_accuracy": 0.8,
                "perturbation_stability": 0.7,
                "help_rate": 0.5,
                "harm_rate": 0.0,
                "bit_allocation_histogram": {"3": 5, "8": 2},
            },
        ],
    )
    spec = table.EvidenceSpec(
        lane="toy",
        method="candidate",
        artifact="results/toy.json",
        evidence_level="test",
        baseline_method="baseline",
        promotion_gate="gate",
    )

    row = table.summarize_spec(spec, tmp_path)

    assert row.status == "present"
    assert row.accuracy == 0.75
    assert row.delta_vs_baseline == 0.5
    assert row.mse == 0.2
    assert row.bytes_proxy == 16
    assert row.atom_recovery == 0.9
    assert row.route_accuracy == 0.8
    assert row.route_stability == 0.7
    assert row.help_rate == 0.5
    assert row.harm_rate == 0.0
    assert row.bit_histogram == "3:5, 8:2"


def test_render_markdown_keeps_missing_rows_visible(tmp_path: Path) -> None:
    spec = table.EvidenceSpec(
        lane="missing lane",
        method="missing_method",
        artifact="results/missing.json",
        evidence_level="blocker",
        baseline_method=None,
        promotion_gate="do not hide",
    )

    rows = table.build_rows(tmp_path, [spec])
    markdown = table.render_markdown(rows)

    assert rows[0].status == "missing"
    assert "artifact missing" in markdown
    assert "missing lane" in markdown
    assert "do not hide" in markdown


def test_default_specs_cover_current_stack_components() -> None:
    methods = {spec.method for spec in table.DEFAULT_SPECS}

    assert "hub_shared_dictionary" in methods
    assert "sticky_paraphrase_stable_routing" in methods
    assert "quant_error_target_bpw_allocator" in methods
    assert "verifier_harm_stop" in methods
    assert "hub_sticky_frontier_verifier_stop" in methods
    assert "oracle_router_control" in methods
    assert "confidence_routing" in methods
