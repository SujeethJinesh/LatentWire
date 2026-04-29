from __future__ import annotations

import json
import pathlib

from scripts.build_source_private_kv_cache_baseline_table import build_kv_cache_baseline_table


def test_kv_cache_baseline_table_accounts_for_endpoint_payloads(tmp_path) -> None:
    config_path = tmp_path / "qwen3_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 1024,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        ),
        encoding="utf-8",
    )
    payload = build_kv_cache_baseline_table(
        endpoint_summaries=[
            pathlib.Path(
                "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json"
            ),
            pathlib.Path(
                "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json"
            ),
        ],
        qwen3_config=config_path,
        output_dir=tmp_path / "kv_cache_baseline",
    )

    assert len(payload["rows"]) == 12
    assert payload["bytes_per_token"]["qjl_1bit_sign_proxy"] == 7168.0
    assert payload["headline"]["packet_payload_bytes"] == [2.0]
    assert payload["headline"]["min_non_packet_qjl_1bit_bytes_vs_packet"] > 10_000
    assert payload["headline"]["min_non_packet_kivi_2bit_bytes_vs_packet"] > 20_000

    comparison = {row["method"]: row for row in payload["baseline_comparison_rows"]}
    assert comparison["LatentWire 2-byte source-private packet"]["source_private"] is True
    assert comparison["KIVI/KVQuant-style low-bit KV cache"]["source_private"] is False

    query_rows = [row for row in payload["rows"] if row["condition"] == "query_aware_diag_span"]
    assert all(row["payload_bytes"] == 14.0 for row in query_rows)
    assert all(row["kv_payload_bytes_qjl_1bit_sign_proxy"] > row["payload_bytes"] for row in query_rows)

    assert (tmp_path / "kv_cache_baseline" / "kv_cache_baseline_table.json").exists()
    assert (tmp_path / "kv_cache_baseline" / "kv_cache_baseline_table.csv").exists()
    assert (tmp_path / "kv_cache_baseline" / "kv_cache_baseline_table.md").exists()
