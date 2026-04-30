from __future__ import annotations

import json

from scripts.summarize_source_private_conditional_pq_innovation_gate import _summarize


def test_conditional_pq_summary_marks_decisive_and_cross_family_rows() -> None:
    rows = [
        {
            "train_family_set": "all",
            "eval_family_set": "all",
            "eval_examples": 500,
            "train_eval_id_intersection_count": 0,
            "budget_bytes": 2,
            "basis_view": "no_diag",
            "pass_gate": True,
            "source_accuracy": 1.0,
            "best_control_accuracy": 0.27,
            "ci95_low_vs_best_control": 0.68,
            "unique_payload_ratio": 0.59,
        },
        {
            "train_family_set": "core",
            "eval_family_set": "holdout",
            "eval_examples": 256,
            "train_eval_id_intersection_count": 0,
            "budget_bytes": 4,
            "basis_view": "shared_text",
            "pass_gate": False,
            "source_accuracy": 0.28,
            "best_control_accuracy": 0.27,
            "ci95_low_vs_best_control": -0.01,
            "unique_payload_ratio": 0.62,
        },
    ]

    summary = _summarize(rows)

    assert summary["pass_gate"] is True
    assert summary["decisive_disjoint_n500_pass_rows"] == 1
    assert summary["less_diagnostic_decisive_pass_rows"] == 1
    assert summary["budget2_decisive_pass_rows"] == 1
    assert summary["cross_family_rows"] == 1
    assert summary["cross_family_pass_rows"] == 0


def test_conditional_pq_summary_json_serializable() -> None:
    payload = _summarize([])

    encoded = json.dumps(payload, sort_keys=True)
    assert "Conditional PQ innovation" in encoded
