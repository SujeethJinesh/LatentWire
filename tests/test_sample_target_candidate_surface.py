from __future__ import annotations

from scripts import sample_target_candidate_surface as sampler


def test_summarize_target_samples_reports_oracle() -> None:
    rows = [
        {"method": "target_sample_s0", "example_id": "a", "correct": False, "normalized_prediction": "1"},
        {"method": "target_sample_s1", "example_id": "a", "correct": True, "normalized_prediction": "2"},
        {"method": "target_sample_s0", "example_id": "b", "correct": False, "normalized_prediction": ""},
        {"method": "target_sample_s1", "example_id": "b", "correct": False, "normalized_prediction": "3"},
    ]

    summary = sampler.summarize(rows)

    assert summary["example_n"] == 2
    assert summary["sample_n"] == 2
    assert summary["candidate_oracle_correct"] == 1
    assert summary["candidate_oracle_ids"] == ["a"]
    assert summary["methods"]["target_sample_s1"]["correct"] == 1
    assert summary["methods"]["target_sample_s0"]["numeric_coverage"] == 1
