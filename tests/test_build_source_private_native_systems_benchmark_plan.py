from __future__ import annotations

import csv

from scripts import build_source_private_native_systems_benchmark_plan as plan


def test_native_systems_plan_has_required_baselines_metrics_and_nonclaims(tmp_path) -> None:
    payload = plan.build_native_systems_plan(output_dir=tmp_path / "out", run_date="2026-05-01")

    assert payload["pass_gate"] is True
    assert payload["headline"]["native_systems_complete"] is False
    assert payload["headline"]["required_metric_count"] >= 20
    assert payload["headline"]["required_baseline_count"] >= 10
    assert payload["headline"]["headline_benchmarks"] == ["ARC-Challenge", "OpenBookQA"]
    assert payload["headline"]["diagnostic_benchmarks"] == ["HellaSwag"]

    baseline_rows = {row["row_id"]: row for row in payload["baseline_rows"]}
    for row_id in (
        "latentwire_packet_cached_source",
        "latentwire_packet_end_to_end_source_scoring",
        "target_only_vllm",
        "target_only_sglang",
        "c2c_cache_to_cache",
        "kvcomm_selective_kv",
        "kvcomm_online_cross_context",
        "qjl_1bit_source_state",
        "turboquant_lowbit_source_state",
    ):
        assert row_id in baseline_rows
    assert baseline_rows["latentwire_packet_cached_source"]["source_private"] is True
    assert baseline_rows["c2c_cache_to_cache"]["source_kv_exposed"] is True

    metric_names = {row["metric"] for row in payload["required_metrics"]}
    for metric in (
        "benchmark",
        "split",
        "commit_hash",
        "gpu_name",
        "cuda_version",
        "driver_version",
        "ttft_ms_p50",
        "tpot_ms_p50",
        "goodput_requests_per_s",
        "peak_gpu_memory_gb",
        "hbm_read_bytes_per_request",
        "pcie_or_nvlink_rx_bytes_per_request",
        "framed_bytes_per_request",
        "transferred_source_state_bytes",
        "source_kv_exposed",
        "wall_time_s",
    ):
        assert metric in metric_names
    assert all(check["pass"] for check in payload["checks"])
    assert any("Do not claim native throughput" in item for item in payload["non_claims"])
    assert any("SSH" in item for item in payload["non_claims"])


def test_native_systems_plan_writes_parseable_artifacts(tmp_path) -> None:
    plan.build_native_systems_plan(output_dir=tmp_path / "out", run_date="2026-05-01")

    assert (tmp_path / "out" / "native_systems_benchmark_plan.json").exists()
    assert (tmp_path / "out" / "native_systems_benchmark_plan.md").exists()
    assert (tmp_path / "out" / "native_systems_baseline_rows.csv").exists()
    assert (tmp_path / "out" / "native_systems_metric_schema.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()

    with (tmp_path / "out" / "native_systems_baseline_rows.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 10
    assert rows[0]["row_id"]
    assert "source_kv_exposed" in rows[0]
