from __future__ import annotations

import csv
import json

from scripts import build_source_private_hellaswag_hidden_innovation_stability_gate as gate


def test_hidden_innovation_stability_gate_promotes_anchored_view(tmp_path) -> None:
    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        split_seeds=(1729, 1731),
        bootstrap_samples=100,
        run_date="2026-05-01",
    )
    headline = payload["headline"]
    diagnostic = payload["unrestricted_model_selection_diagnostic"]

    assert payload["pass_gate"] is True
    assert headline["pass_count"] == 2
    assert headline["split_seed_count"] == 2
    assert headline["selected_view_counts"] == {"score_hidden_residual": 2}
    assert headline["delta_vs_best_label_copy_min"] >= 0.02
    assert headline["paired_ci95_low_vs_best_label_copy_min"] > 0
    assert headline["selected_minus_zero_hidden_control_min"] >= 0.02
    assert diagnostic["pass_count"] < diagnostic["split_seed_count"]
    assert "hidden_residual_only" in diagnostic["selected_view_counts"]
    assert payload["packet_contract"]["raw_payload_bytes"] == 2
    assert payload["packet_contract"]["framed_record_bytes"] == 5
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["packet_contract"]["source_kv_exposed"] is False
    assert payload["packet_contract"]["raw_hidden_vector_transmitted"] is False
    assert payload["packet_contract"]["raw_scores_transmitted"] is False


def test_hidden_innovation_stability_gate_writes_artifacts(tmp_path) -> None:
    gate.build_gate(
        output_dir=tmp_path / "out",
        split_seeds=(1729, 1731),
        bootstrap_samples=100,
        run_date="2026-05-01",
    )

    summary = tmp_path / "out" / "hellaswag_hidden_innovation_stability_gate.json"
    rows = tmp_path / "out" / "stability_rows.csv"
    assert summary.exists()
    assert rows.exists()
    assert (tmp_path / "out" / "hellaswag_hidden_innovation_stability_gate.md").exists()
    assert (tmp_path / "out" / "candidate_readouts.jsonl").exists()
    assert (tmp_path / "out" / "manifest.json").exists()

    parsed = json.loads(summary.read_text(encoding="utf-8"))
    assert parsed["pass_gate"] is True
    with rows.open(encoding="utf-8", newline="") as handle:
        parsed_rows = list(csv.DictReader(handle))
    assert len(parsed_rows) == 2
    assert {row["selection_policy"] for row in parsed_rows} == {"anchored_score_hidden_residual"}
