import json

from scripts import summarize_reachability_union as union


def test_summarize_reachability_union_merges_ids(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "reference_n": 4,
                "sample_oracle_ids": ["a", "b"],
                "sample_oracle_correct": 2,
                "c2c_clean_residual_in_pool_ids": ["a"],
                "c2c_clean_residual_total": 2,
                "c2c_teacher_only_in_pool_ids": ["b"],
                "c2c_teacher_only_total": 3,
                "source_contrastive_clean_in_pool_ids": [],
                "source_contrastive_clean_total": 1,
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "reference_n": 4,
                "sample_oracle_ids": ["b", "c"],
                "sample_oracle_correct": 2,
                "c2c_clean_residual_in_pool_ids": ["c"],
                "c2c_clean_residual_total": 2,
                "c2c_teacher_only_in_pool_ids": ["b", "c"],
                "c2c_teacher_only_total": 3,
                "source_contrastive_clean_in_pool_ids": ["d"],
                "source_contrastive_clean_total": 1,
            }
        ),
        encoding="utf-8",
    )

    payload = union.main(
        [
            "--reachability",
            f"first={first}",
            "--reachability",
            f"second={second}",
            "--date",
            "2026-04-27",
            "--output-json",
            str(tmp_path / "union.json"),
            "--output-md",
            str(tmp_path / "union.md"),
        ]
    )

    assert payload["sample_oracle_ids"] == ["a", "b", "c"]
    assert payload["sample_oracle_correct"] == 3
    assert payload["c2c_clean_residual_in_pool_ids"] == ["a", "c"]
    assert payload["c2c_teacher_only_in_pool_ids"] == ["b", "c"]
    assert payload["source_contrastive_clean_in_pool_ids"] == ["d"]
