from __future__ import annotations

import csv
import json

from scripts.build_source_private_anti_lookup_label_blind_summary import build_anti_lookup_summary


def test_anti_lookup_label_blind_summary_passes_collapse_smoke(tmp_path) -> None:
    payload = build_anti_lookup_summary(
        label_blind_summaries=[
            "results/source_private_anti_lookup_label_blind_20260429/core_seed29_qwen3_n8_label_blind/summary.json",
            "results/source_private_anti_lookup_label_blind_20260429/holdout_seed30_qwen3_n8_label_blind/summary.json",
            "results/source_private_anti_lookup_label_blind_20260429/core_seed29_qwen3_n32_label_blind/summary.json",
            "results/source_private_anti_lookup_label_blind_20260429/holdout_seed30_qwen3_n32_label_blind/summary.json",
        ],
        positive_summaries=[
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
        ],
        output_dir=tmp_path,
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["rows"] == 4
    assert payload["headline"]["collapse_pass_rows"] == 4
    assert payload["headline"]["max_opaque_minus_target"] <= 0.05
    assert payload["headline"]["max_opaque_ci95_high_vs_target"] <= 0.10
    assert payload["headline"]["max_opaque_strict_ci95_high_vs_target"] <= 0.10
    assert payload["headline"]["min_diagnostic_table_positive_lift"] >= 0.15
    assert payload["headline"]["all_exact_id_parity"] is True
    assert (tmp_path / "anti_lookup_label_blind_summary.json").exists()
    assert (tmp_path / "anti_lookup_label_blind_summary.csv").exists()
    assert (tmp_path / "anti_lookup_label_blind_summary.md").exists()
    assert (tmp_path / "manifest.json").exists()


def test_anti_lookup_label_blind_summary_writes_valid_outputs(tmp_path) -> None:
    build_anti_lookup_summary(
        label_blind_summaries=[
            "results/source_private_anti_lookup_label_blind_20260429/core_seed29_qwen3_n8_label_blind/summary.json",
            "results/source_private_anti_lookup_label_blind_20260429/holdout_seed30_qwen3_n8_label_blind/summary.json",
            "results/source_private_anti_lookup_label_blind_20260429/core_seed29_qwen3_n32_label_blind/summary.json",
            "results/source_private_anti_lookup_label_blind_20260429/holdout_seed30_qwen3_n32_label_blind/summary.json",
        ],
        positive_summaries=[
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
        ],
        output_dir=tmp_path,
    )
    summary = json.loads((tmp_path / "anti_lookup_label_blind_summary.json").read_text(encoding="utf-8"))
    with (tmp_path / "anti_lookup_label_blind_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert summary["pass_gate"] is True
    assert len(rows) == 4
    assert rows[0]["candidate_view"] == "label_blind"


def test_anti_lookup_label_blind_summary_passes_n160_scaleup(tmp_path) -> None:
    payload = build_anti_lookup_summary(
        label_blind_summaries=[
            "results/source_private_anti_lookup_label_blind_20260430/core_seed29_qwen3_n160_label_blind/summary.json",
            "results/source_private_anti_lookup_label_blind_20260430/holdout_seed30_qwen3_n160_label_blind/summary.json",
        ],
        positive_summaries=[
            "results/source_private_mac_endpoint_proxy_frontier_20260429/core_seed29_qwen3_n160_cpu_label_strict_controls/summary.json",
            "results/source_private_mac_endpoint_proxy_frontier_20260429/holdout_seed30_qwen3_n160_cpu_label_strict_controls/summary.json",
        ],
        output_dir=tmp_path,
    )

    assert payload["pass_gate"] is True
    assert payload["scale_rung"] == "medium anti-lookup stress"
    assert payload["headline"]["rows"] == 2
    assert payload["headline"]["collapse_pass_rows"] == 2
    assert payload["headline"]["max_opaque_minus_target"] == 0.0
    assert payload["headline"]["max_opaque_ci95_high_vs_target"] == 0.0
    assert payload["headline"]["min_diagnostic_table_positive_lift"] >= 0.425
    assert "n=500 deterministic label-blind stress" in payload["next_gate"]
