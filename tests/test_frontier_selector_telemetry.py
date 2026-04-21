from __future__ import annotations

import json

from scripts import analyze_frontier_selector_telemetry as telemetry


def test_normalize_existing_frontier_aliases() -> None:
    row = {
        "method": "exact_patch_effect_protect",
        "patch_rank_correlation": 1.0,
        "feature_overlap_persistence": 0.42,
        "protected_atoms": 4,
        "help_vs_prune_uniform_quant": 0.125,
        "harm_vs_prune_uniform_quant": 0.0,
        "missed_help_rate": 0.03125,
        "false_prune_rate": 0.015625,
        "bytes_proxy": 88.5,
        "compute_proxy": 681.0,
        "selector_stability": 0.75,
    }

    normalized = telemetry.normalize_row(row, input_name="toy.json", config={"low_bits": 3, "high_bits": 8})

    assert normalized["selector_method"] == "exact_patch_effect_protect"
    assert normalized["patch_corr"] == 1.0
    assert normalized["quant_error_corr"] is None
    assert normalized["feature_persistence"] == 0.42
    assert normalized["protected_ids"] == []
    assert normalized["bit_allocation"] == {
        "low_bits": 3,
        "high_bits": 8,
        "protected_count": 4,
        "source": "count_only",
    }
    assert normalized["help"] == 0.125
    assert normalized["harm"] == 0.0
    assert normalized["missed_help"] == 0.03125
    assert normalized["false_prune"] == 0.015625
    assert normalized["bytes"] == 88.5
    assert normalized["compute"] == 681.0
    assert normalized["stability"] == 0.75


def test_load_and_summarize_payload_rows(tmp_path) -> None:
    source = tmp_path / "frontier.json"
    source.write_text(
        json.dumps(
            {
                "config": {"low_bits": 2, "high_bits": 8},
                "rows": [
                    {
                        "method": "quant_error_protect",
                        "patch_rank_correlation": 0.27,
                        "quant_error_rank_correlation": 0.91,
                        "feature_overlap_persistence": 0.2,
                        "protected_ids": [1, 3],
                        "help_vs_prune_uniform_quant": 0.1,
                        "harm_vs_prune_uniform_quant": 0.0,
                        "missed_help_rate": 0.05,
                        "false_prune_rate": 0.02,
                        "bytes_proxy": 90.0,
                        "compute_proxy": 100.0,
                        "selector_stability": 0.8,
                    },
                    {
                        "method": "random_protect",
                        "patch_rank_correlation": -0.1,
                        "quant_error_rank_correlation": -0.2,
                        "feature_overlap_persistence": 0.05,
                        "protected_ids": [5, 7],
                        "help_vs_prune_uniform_quant": 0.0,
                        "harm_vs_prune_uniform_quant": 0.1,
                        "missed_help_rate": 0.4,
                        "false_prune_rate": 0.2,
                        "bytes_proxy": 90.0,
                        "compute_proxy": 100.0,
                        "selector_stability": 0.1,
                    },
                ],
            }
        )
    )

    payload = telemetry.build_payload([source])

    assert payload["summary"]["row_count"] == 2
    assert payload["summary"]["schema_fields"] == list(telemetry.SCHEMA_FIELDS)
    assert payload["summary"]["best_patch_corr_selector"] == "quant_error_protect"
    assert payload["summary"]["best_help_selector"] == "quant_error_protect"
    assert payload["rows"][0]["source"] == str(source)
    assert payload["rows"][0]["protected_ids"] == [1, 3]


def test_cli_writes_fixture_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "frontier_selector_telemetry.json"
    markdown = tmp_path / "frontier_selector_telemetry.md"

    payload = telemetry.main(["--output", str(output), "--output-md", str(markdown)])
    loaded = json.loads(output.read_text())

    assert loaded == payload
    assert loaded["inputs"] == ["embedded_fixture"]
    assert loaded["summary"]["row_count"] == 2
    assert set(telemetry.SCHEMA_FIELDS).issubset(loaded["rows"][0])
    assert "# Frontier Selector Telemetry" in markdown.read_text()
    assert "| Selector | Patch corr | Quant-error corr | Feature persistence | Protected ids |" in markdown.read_text()
