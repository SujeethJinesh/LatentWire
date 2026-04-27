import json

from scripts import compare_candidate_pool_reachability as compare


def test_compare_reports_new_and_lost_ids(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            {
                "reference_n": 4,
                "sample_oracle_ids": ["a", "b"],
                "sample_oracle_correct": 2,
                "c2c_clean_residual_in_pool_ids": ["a"],
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "reference_n": 4,
                "sample_oracle_ids": ["b", "c"],
                "sample_oracle_correct": 2,
                "c2c_clean_residual_in_pool_ids": ["c"],
                "c2c_clean_residual_in_pool": 1,
                "c2c_clean_residual_total": 2,
            }
        ),
        encoding="utf-8",
    )

    payload = compare.main(
        [
            "--baseline-reachability",
            str(baseline),
            "--candidate-reachability",
            str(candidate),
            "--date",
            "2026-04-27",
            "--output-json",
            str(tmp_path / "out.json"),
            "--output-md",
            str(tmp_path / "out.md"),
        ]
    )

    assert payload["new_oracle_ids"] == ["c"]
    assert payload["lost_oracle_ids"] == ["a"]
    assert payload["new_c2c_clean_residual_ids"] == ["c"]
    assert payload["decision"].startswith("Pass")
