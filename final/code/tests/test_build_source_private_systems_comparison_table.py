from __future__ import annotations

from scripts.build_source_private_systems_comparison_table import (
    DEFAULT_HELDOUT_GATE,
    DEFAULT_KV_TABLE,
    DEFAULT_LEARNED_GATE,
    DEFAULT_QJL_SUMMARY,
    build_systems_comparison_table,
)


def test_systems_comparison_table_collects_headline_and_caveat_rows(tmp_path) -> None:
    payload = build_systems_comparison_table(
        learned_gate=DEFAULT_LEARNED_GATE,
        heldout_gate=DEFAULT_HELDOUT_GATE,
        qjl_summary=DEFAULT_QJL_SUMMARY,
        kv_table=DEFAULT_KV_TABLE,
        output_dir=tmp_path / "systems_comparison",
    )

    assert payload["gate"] == "source_private_systems_comparison_table"
    assert payload["headline"]["headline_learned_pass_rows"] == 3
    assert payload["headline"]["same_surface_text_max_delta_vs_target"] == 0.0
    assert payload["headline"]["compression_scalar_accuracy"] == 1.0
    assert payload["headline"]["compression_qjl_accuracy"] == 1.0
    assert payload["headline"]["min_endpoint_nonpacket_qjl_1bit_bytes_vs_packet"] > 10_000

    rows = payload["rows"]
    assert any(row["row_group"] == "heldout_boundary" and row["pass_rule_result"] == "fail" for row in rows)
    assert any(row["method"] == "same-byte structured text relay" for row in rows)
    assert any(row["method"] == "QJL-style residual projection" for row in rows)
    assert any(row["method"] == "query_aware_diag_span" for row in rows)
    assert (tmp_path / "systems_comparison" / "source_private_systems_comparison_table.json").exists()
    assert (tmp_path / "systems_comparison" / "source_private_systems_comparison_table.csv").exists()
    assert (tmp_path / "systems_comparison" / "source_private_systems_comparison_table.md").exists()
