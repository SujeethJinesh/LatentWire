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
