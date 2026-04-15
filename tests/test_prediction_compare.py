from __future__ import annotations

import json

import pytest

from latent_bridge import prediction_compare


def _write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_compare_prediction_records_reports_cross_file_flips() -> None:
    candidate = [
        {"index": 0, "method": "rotalign_kv_gate_0.05", "correct": True},
        {"index": 1, "method": "rotalign_kv_gate_0.05", "correct": True},
        {"index": 2, "method": "rotalign_kv_gate_0.05", "correct": False},
    ]
    baseline = [
        {"index": 0, "method": "rotalign_kv_gate_0.05", "correct": False},
        {"index": 1, "method": "rotalign_kv_gate_0.05", "correct": True},
        {"index": 2, "method": "rotalign_kv_gate_0.05", "correct": True},
    ]

    row = prediction_compare.compare_prediction_records(
        candidate,
        baseline,
        method="rotalign_kv_gate_0.05",
        candidate_label="real",
        baseline_label="zero",
        n_bootstrap=32,
    )

    assert row["candidate_accuracy"] == pytest.approx(2 / 3)
    assert row["baseline_accuracy"] == pytest.approx(2 / 3)
    assert row["method_only"] == 1.0
    assert row["baseline_only"] == 1.0
    assert row["both_correct"] == 1.0


def test_compare_prediction_records_supports_different_baseline_method() -> None:
    candidate = [
        {"index": 0, "method": "rotalign_kv_gate_0.25", "correct": True},
        {"index": 1, "method": "rotalign_kv_gate_0.25", "correct": False},
    ]
    baseline = [
        {"index": 0, "method": "rotalign_kv_gate_0.05", "correct": True},
        {"index": 1, "method": "rotalign_kv_gate_0.05", "correct": True},
    ]

    row = prediction_compare.compare_prediction_records(
        candidate,
        baseline,
        method="rotalign_kv_gate_0.25",
        baseline_method="rotalign_kv_gate_0.05",
        n_bootstrap=32,
    )

    assert row["candidate_method"] == "rotalign_kv_gate_0.25"
    assert row["baseline_method"] == "rotalign_kv_gate_0.05"
    assert row["delta_accuracy"] == -0.5


def test_compare_prediction_files_filters_common_methods_by_prefix(tmp_path) -> None:
    candidate_path = tmp_path / "real.jsonl"
    baseline_path = tmp_path / "zero.jsonl"
    _write_jsonl(
        candidate_path,
        [
            {"index": 0, "method": "target_alone", "correct": True},
            {"index": 0, "method": "rotalign_kv_gate_0.05", "correct": True},
            {"index": 0, "method": "text_to_text", "correct": False},
        ],
    )
    _write_jsonl(
        baseline_path,
        [
            {"index": 0, "method": "target_alone", "correct": True},
            {"index": 0, "method": "rotalign_kv_gate_0.05", "correct": False},
            {"index": 0, "method": "text_to_text", "correct": False},
        ],
    )

    rows = prediction_compare.compare_prediction_files(
        candidate_path,
        baseline_path,
        method_prefix="rotalign_kv_gate_",
        candidate_label="real",
        baseline_label="zero",
        n_bootstrap=32,
    )

    assert [row["method"] for row in rows] == ["rotalign_kv_gate_0.05"]
    assert rows[0]["delta_accuracy"] == 1.0


def test_format_markdown_includes_core_stats() -> None:
    md = prediction_compare.format_markdown(
        [
            {
                "method": "rotalign_kv_gate_0.05",
                "candidate_accuracy": 0.5,
                "baseline_accuracy": 0.4,
                "delta_accuracy": 0.1,
                "method_only": 2.0,
                "baseline_only": 1.0,
                "bootstrap_delta_low": -0.1,
                "bootstrap_delta_high": 0.3,
                "mcnemar_p": 0.5,
            }
        ]
    )

    assert "rotalign_kv_gate_0.05" in md
    assert "+0.1000" in md
