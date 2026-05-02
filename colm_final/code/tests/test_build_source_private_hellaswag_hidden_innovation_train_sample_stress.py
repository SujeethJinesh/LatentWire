from __future__ import annotations

import csv
import json

from scripts import build_source_private_hellaswag_hidden_innovation_train_sample_stress as stress


def test_train_sample_stress_original_sample_is_not_enough_for_promotion(tmp_path) -> None:
    payload = stress.build_gate(
        output_dir=tmp_path / "out",
        train_sample_seeds=(1729,),
        split_seeds=(1729, 1731),
        bootstrap_samples=100,
        run_date="2026-05-01",
    )
    headline = payload["headline"]

    assert payload["pass_gate"] is False
    assert headline["new_train_sample_seed_count"] == 0
    assert headline["pass_count"] == 2
    assert headline["selected_view_counts"] == {"score_hidden_residual": 2}
    assert payload["packet_contract"]["raw_payload_bytes"] == 2
    assert payload["packet_contract"]["framed_record_bytes"] == 5
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["packet_contract"]["source_kv_exposed"] is False
    assert payload["packet_contract"]["raw_hidden_vector_transmitted"] is False
    assert payload["packet_contract"]["raw_scores_transmitted"] is False


def test_train_sample_stress_writes_artifacts(tmp_path) -> None:
    stress.build_gate(
        output_dir=tmp_path / "out",
        train_sample_seeds=(1729,),
        split_seeds=(1729, 1731),
        bootstrap_samples=100,
        run_date="2026-05-01",
    )

    summary = tmp_path / "out" / "hellaswag_hidden_innovation_train_sample_stress.json"
    rows = tmp_path / "out" / "stress_rows.csv"
    assert summary.exists()
    assert rows.exists()
    assert (tmp_path / "out" / "hellaswag_hidden_innovation_train_sample_stress.md").exists()
    assert (tmp_path / "out" / "sample_caches.jsonl").exists()
    assert (tmp_path / "out" / "candidate_readouts.jsonl").exists()
    assert (tmp_path / "out" / "manifest.json").exists()

    parsed = json.loads(summary.read_text(encoding="utf-8"))
    assert parsed["pass_gate"] is False
    with rows.open(encoding="utf-8", newline="") as handle:
        parsed_rows = list(csv.DictReader(handle))
    assert len(parsed_rows) == 2
    assert {row["train_sample_seed"] for row in parsed_rows} == {"1729"}
