from __future__ import annotations

from scripts.build_source_private_protocol_stress_table import build_protocol_stress_table


def test_protocol_stress_table_tracks_remap_evidence_and_open_gap(tmp_path) -> None:
    payload = build_protocol_stress_table(output_dir=tmp_path / "protocol")
    stressors = {row["stressor"] for row in payload["rows"]}
    statuses = {row["status"] for row in payload["rows"]}

    assert payload["headline"]["total_rows"] == len(payload["rows"])
    assert "diagnostic codebook remap" in stressors
    assert "slot-feature remap" in stressors
    assert "canonical candidate-order remap" in stressors
    assert "pass" in statuses
    assert "near-miss" in statuses
    assert "prompt paraphrases" in payload["open_gap"]
    assert (tmp_path / "protocol" / "protocol_stress_table.md").exists()
