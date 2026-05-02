from __future__ import annotations

from scripts import run_gsm8k_smoke_contract as smoke


def test_method_grouping_and_pairing_helpers() -> None:
    target_records = [
        {"method": "target_alone", "example_id": "a", "correct": True, "prediction": "18", "generated_tokens": 3, "latency_sec": 1.0},
        {"method": "target_alone", "example_id": "b", "correct": False, "prediction": "foo", "generated_tokens": 4, "latency_sec": 2.0},
    ]
    rotalign_records = [
        {"method": "rotalign_kv_gate_0.10", "example_id": "a", "correct": True, "prediction": "18", "generated_tokens": 5, "latency_sec": 1.5},
        {"method": "rotalign_kv_gate_0.10", "example_id": "b", "correct": True, "prediction": "7", "generated_tokens": 6, "latency_sec": 1.5},
    ]
    grouped = smoke._group_by_method(target_records + rotalign_records)
    assert set(grouped) == {"target_alone", "rotalign_kv"}

    paired = smoke._paired_vs_baseline(grouped["rotalign_kv"], grouped["target_alone"])
    assert paired == {"win": 1, "loss": 0, "tie": 1}


def test_method_row_tracks_numeric_coverage_and_empty_predictions() -> None:
    row = smoke._method_row(
        [
            {"method": "c2c", "example_id": "a", "correct": True, "prediction": "The answer is 18", "generated_tokens": 3, "latency_sec": 1.0},
            {"method": "c2c", "example_id": "b", "correct": False, "prediction": " ", "generated_tokens": 5, "latency_sec": 3.0},
        ]
    )
    assert row["n"] == 2
    assert row["accuracy"] == 0.5
    assert row["generated_tokens_avg"] == 4.0
    assert row["latency_sec_avg"] == 2.0
    assert row["numeric_extraction_coverage"] == 1
    assert row["empty_predictions"] == 1
