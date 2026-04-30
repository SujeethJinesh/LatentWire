from __future__ import annotations

from scripts.build_source_private_pq_systems_comparison_table import (
    DEFAULT_DECODE_FRONTIER,
    DEFAULT_GEOMETRY_STRESS,
    DEFAULT_PACKET_TRACE_CARD,
    DEFAULT_PQ_GATE,
    DEFAULT_RATE_FRONTIER,
    DEFAULT_VERIFIER_TRACE,
    build_pq_systems_comparison_table,
)


def test_pq_systems_comparison_table_collects_latest_rows(tmp_path) -> None:
    payload = build_pq_systems_comparison_table(
        pq_gate=DEFAULT_PQ_GATE,
        geometry_stress=DEFAULT_GEOMETRY_STRESS,
        decode_frontier=DEFAULT_DECODE_FRONTIER,
        verifier_trace=DEFAULT_VERIFIER_TRACE,
        packet_trace_card=DEFAULT_PACKET_TRACE_CARD,
        rate_frontier=DEFAULT_RATE_FRONTIER,
        output_dir=tmp_path / "pq_systems_table",
    )

    assert payload["gate"] == "source_private_pq_systems_comparison_table"
    assert payload["pass_gate"] is True
    assert payload["headline"]["pq_geometry_rows"] == 4
    assert payload["headline"]["pq_mitigation_rows"] == 3
    assert payload["headline"]["pq_min_delta_vs_best_control"] >= 0.21
    assert payload["headline"]["protected_hadamard_unique_payloads_min"] <= 405
    assert payload["headline"]["same_byte_text_accuracy_max"] == 0.25
    assert payload["headline"]["query_aware_text_raw_ratio"] == 7.0
    assert payload["headline"]["kv_raw_ratio_min"] > 10_000

    methods = {row["method"] for row in payload["rows"]}
    expected = {
        "canonical 4-byte product-codebook packet",
        "utility-OPQ product-codebook packet",
        "protected Hadamard product-codebook packet",
        "utility-protected Hadamard product-codebook packet",
        "scalar Wyner-Ziv residual packet",
        "frozen Qwen3 binary-verifier packet",
        "same-byte structured text",
        "query-aware structured text oracle",
        "full hidden-log relay",
        "QJL-style 1-bit source KV byte floor",
        "KIVI/KVQuant-style 2-bit source KV byte floor",
        "C2C cache-to-cache communication",
        "KVComm / KVCOMM selective KV communication",
    }
    assert expected <= methods

    kv_rows = [row for row in payload["rows"] if row["row_group"] == "kv_byte_floor"]
    assert kv_rows
    assert all(row["source_kv_exposed"] is True for row in kv_rows)
    assert all(row["pass_rule_result"] == "accounting_contrast" for row in kv_rows)

    external_rows = [row for row in payload["rows"] if row["row_group"] == "external_reference"]
    assert external_rows
    assert all(row["pass_rule_result"] == "reference_only" for row in external_rows)

    assert (tmp_path / "pq_systems_table" / "source_private_pq_systems_comparison_table.json").exists()
    assert (tmp_path / "pq_systems_table" / "source_private_pq_systems_comparison_table.csv").exists()
    assert (tmp_path / "pq_systems_table" / "source_private_pq_systems_comparison_table.md").exists()
