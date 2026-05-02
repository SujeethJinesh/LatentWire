from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_hidden_innovation_repair_probe as probe


def test_hidden_innovation_probe_clears_label_copy_and_controls(tmp_path) -> None:
    payload = probe.build_probe(output_dir=tmp_path / "out", run_date="2026-05-01")
    headline = payload["headline"]

    assert payload["pass_gate"] is True
    assert headline["selected_view"] == "score_hidden_residual"
    assert headline["selected_eval_accuracy"] >= 0.49
    assert headline["selected_minus_best_label_copy"] >= 0.02
    assert headline["paired_ci95_selected_vs_best_label_copy"]["ci95_low"] > 0
    assert headline["selected_minus_zero_hidden_control"] >= 0.02
    assert headline["wrong_example_hidden_control_accuracy"] < headline["best_label_copy_eval_accuracy"]
    assert headline["candidate_roll_hidden_control_accuracy"] < headline["best_label_copy_eval_accuracy"]
    assert payload["packet_contract"]["raw_payload_bytes"] == 2
    assert payload["packet_contract"]["framed_record_bytes"] == 5
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["packet_contract"]["source_kv_exposed"] is False
    assert payload["packet_contract"]["raw_hidden_vector_transmitted"] is False
    assert payload["packet_contract"]["raw_scores_transmitted"] is False


def test_hidden_innovation_probe_writes_artifacts(tmp_path) -> None:
    probe.build_probe(output_dir=tmp_path / "out", run_date="2026-05-01")

    summary = tmp_path / "out" / "hellaswag_hidden_innovation_repair_probe.json"
    assert summary.exists()
    assert (tmp_path / "out" / "hellaswag_hidden_innovation_repair_probe.md").exists()
    assert (tmp_path / "out" / "candidate_readouts.jsonl").exists()
    assert (tmp_path / "out" / "predictions.jsonl").exists()
    assert (tmp_path / "out" / "manifest.json").exists()

    parsed = json.loads(summary.read_text(encoding="utf-8"))
    assert parsed["pass_gate"] is True
    assert parsed["headline"]["source_top2_oracle_accuracy"] > parsed["headline"]["selected_eval_accuracy"]
