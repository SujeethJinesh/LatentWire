from __future__ import annotations

import json

from scripts.summarize_source_private_conditional_pq_basis_schema_grid import _summarize


def test_basis_schema_grid_summary_detects_no_bidirectional_pass() -> None:
    records = [
        {
            "mode": "plausible_decoys",
            "basis": "semantic",
            "direction": "core_to_holdout",
            "pass": True,
            "source": 0.45,
            "target": 0.25,
            "best_control": 0.26,
            "ci95_low": 0.12,
            "unquantized": 0.60,
        },
        {
            "mode": "plausible_decoys",
            "basis": "semantic",
            "direction": "holdout_to_core",
            "pass": False,
            "source": 0.25,
            "target": 0.25,
            "best_control": 0.25,
            "ci95_low": 0.0,
            "unquantized": 0.50,
        },
    ]

    summary = _summarize(records)

    assert summary["pass_gate"] is False
    assert summary["pass_rows"] == 1
    assert summary["bidirectional_pass_count"] == 0
    assert abs(summary["max_source_minus_best_control"] - 0.19) < 1e-9


def test_basis_schema_grid_summary_json_serializable() -> None:
    payload = _summarize([])

    encoded = json.dumps(payload, sort_keys=True)
    assert "static-basis" in encoded
