from __future__ import annotations

import csv
import json

from scripts.build_source_private_systems_caveat_frontier import build_systems_caveat_frontier


def test_systems_caveat_frontier_passes_with_endpoint_rows(tmp_path) -> None:
    payload = build_systems_caveat_frontier(
        endpoint_summaries=[
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
        ],
        uncertainty_summaries=[
            "results/source_private_endpoint_uncertainty_20260429/core_label_strict_n160/summary.json",
            "results/source_private_endpoint_uncertainty_20260429/label_strict_n160/summary.json",
        ],
        terse_failure_summary="results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n16_cpu_terse/summary.json",
        kv_table_path="results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json",
        output_dir=tmp_path,
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["passing_endpoint_rows"] == 2
    assert payload["headline"]["packet_payload_bytes"] == 2.0
    assert payload["headline"]["min_packet_minus_target_accuracy"] >= 0.15
    assert payload["headline"]["min_packet_minus_best_control_accuracy"] >= 0.15
    assert payload["headline"]["min_packet_vs_query_payload_compression"] >= 7.0
    assert payload["headline"]["min_qjl_1bit_cache_bytes_vs_packet"] >= 1000.0
    assert payload["headline"]["terse_prompt_pass_gate"] is False

    assert (tmp_path / "systems_caveat_frontier.json").exists()
    assert (tmp_path / "systems_caveat_frontier.csv").exists()
    assert (tmp_path / "systems_caveat_frontier.md").exists()
    assert (tmp_path / "manifest.json").exists()


def test_systems_caveat_frontier_records_non_claims_and_related_work(tmp_path) -> None:
    payload = build_systems_caveat_frontier(
        endpoint_summaries=[
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
        ],
        uncertainty_summaries=[
            "results/source_private_endpoint_uncertainty_20260429/core_label_strict_n160/summary.json",
            "results/source_private_endpoint_uncertainty_20260429/label_strict_n160/summary.json",
        ],
        terse_failure_summary="results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n16_cpu_terse/summary.json",
        kv_table_path="results/source_private_kv_cache_baseline_table_20260429/kv_cache_baseline_table.json",
        output_dir=tmp_path,
    )

    related = {row["method"]: row for row in payload["related_work"]}
    assert "TurboQuant" in related
    assert "QJL" in related
    assert "C2C cache-to-cache communication" in related
    assert any("No production GPU serving throughput claim" in item for item in payload["non_claims"])
    assert any(row["pass_gate"] is False and row["prompt_contract"] == "terse" for row in payload["rows"])

    with (tmp_path / "systems_caveat_frontier.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((tmp_path / "systems_caveat_frontier.json").read_text(encoding="utf-8"))
    assert len(rows) == 3
    assert summary["pass_gate"] is True
