from __future__ import annotations

from scripts.build_source_private_cpu_systems_frontier import build_cpu_frontier


def test_cpu_systems_frontier_includes_passes_and_failures(tmp_path) -> None:
    payload = build_cpu_frontier(output_dir=tmp_path / "frontier")
    contributions = {row["contribution"] for row in payload["rows"]}
    statuses = {row["status"] for row in payload["rows"]}

    assert payload["headline"]["total_rows"] == len(payload["rows"])
    assert "byte-rate systems frontier" in contributions
    assert "learned scalar packet" in contributions
    assert "canonical RASP cross-family falsification" in contributions
    assert "consistency posterior negative ablation" in contributions
    assert "Mac endpoint-proxy byte/TTFT frontier" in contributions
    assert "pass" in statuses
    assert "fail" in statuses
    assert (tmp_path / "frontier" / "cpu_systems_frontier.csv").exists()

    endpoint_rows = {row["row_id"]: row for row in payload["rows"]}
    strict_core = endpoint_rows["endpoint_proxy_core_n32_audit_strict_controls"]
    strict_holdout = endpoint_rows["endpoint_proxy_holdout_n32_audit_strict_controls"]
    assert payload["headline"]["total_rows"] >= 84
    assert strict_core["status"] == "pass"
    assert strict_core["accuracy"] > strict_core["best_control_accuracy"]
    assert strict_core["best_control_accuracy"] == 0.28125
    assert strict_holdout["status"] == "pass"
    assert strict_holdout["accuracy"] > strict_holdout["best_control_accuracy"]
    assert strict_holdout["best_control_accuracy"] == 0.3125
