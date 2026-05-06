import json
from pathlib import Path

from experimental.sinkkv.phase2.sinkkv_deterministic_probe import run_probe


def test_sinkkv_deterministic_probe_writes_required_packet(tmp_path: Path) -> None:
    summary = run_probe(output_dir=tmp_path)

    assert summary["seed"] == 20260506
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "raw_rows.jsonl").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "decision.md").exists()

    rows = {row["row"]: row for row in summary["rows"]}
    for required in [
        "full_precision_kv",
        "uniform_mxfp4_kv",
        "sink_protected_budget_matched_kv",
        "recent_protected_budget_matched_kv",
    ]:
        assert required in rows

    assert rows["full_precision_kv"]["attention_output_rel_l2"] == 0.0
    assert abs(rows["uniform_mxfp4_kv"]["budget_bits_per_element"] - 4.0) < 1e-8
    assert abs(rows["sink_protected_budget_matched_kv"]["budget_bits_per_element"] - 4.0) < 1e-8
    assert abs(rows["recent_protected_budget_matched_kv"]["budget_bits_per_element"] - 4.0) < 1e-8

    markdown = (tmp_path / "summary.md").read_text()
    for phrase in ["not GPU speed", "not benchmark accuracy", "does not skip QK_sink"]:
        assert phrase in markdown

    on_disk = json.loads((tmp_path / "summary.json").read_text())
    assert on_disk["decision"] == summary["decision"]
