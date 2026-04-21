from __future__ import annotations

import json

from scripts import analyze_tokenizer_stress_split as stress


def test_tokenizer_stress_split_is_deterministic_and_covers_categories() -> None:
    config = stress.TokenizerStressSplitConfig(seed=3, remap_capacity=6)
    examples = stress._default_examples()

    payload = stress.run_analysis(config, examples)
    payload_again = stress.run_analysis(config, examples)

    assert payload == payload_again
    assert payload["summary"]["example_count"] == len(examples)
    assert payload["summary"]["source_target_boundary_f1"] < 1.0
    assert payload["summary"]["byte_span_remap_coverage"] > 0.0
    assert payload["summary"]["token_id_exact_reconstruction"] < 1.0
    assert payload["summary"]["byte_span_exact_reconstruction"] == 1.0

    categories = {row["category"] for row in payload["rows"]}
    assert {
        "overall",
        "unicode",
        "math_units",
        "decimals",
        "variables",
        "punctuation",
        "multi_byte_span",
    }.issubset(categories)

    by_category = {row["category"]: row for row in payload["rows"]}
    assert by_category["multi_byte_span"]["multi_byte_source_span_rate"] > 0.0
    assert by_category["decimals"]["example_count"] >= 2
    assert by_category["punctuation"]["source_target_boundary_f1"] <= 1.0

    for row in payload["examples"]:
        assert row["byte_count"] >= row["char_count"]
        assert 0.0 <= row["source_target_boundary_f1"] <= 1.0
        assert 0.0 <= row["byte_span_remap_coverage"] <= 1.0
        assert row["target_regroup_exact_reconstruction"] == 1.0


def test_tokenizer_stress_split_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "tokenizer_stress_split.json"
    markdown = tmp_path / "tokenizer_stress_split.md"

    payload = stress.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "4",
            "--remap-capacity",
            "5",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 4
    assert on_disk["config"]["remap_capacity"] == 5
    assert len(on_disk["examples"]) == 12
    assert "Tokenizer Stress Split" in markdown.read_text()
