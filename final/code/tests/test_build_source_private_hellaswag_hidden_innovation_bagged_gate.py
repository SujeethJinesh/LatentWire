from __future__ import annotations

import csv
import json

from scripts import build_source_private_hellaswag_hidden_innovation_bagged_gate as bagged


def test_hybrid_vote_on_score_agreement_uses_vote_only_on_score_collapse() -> None:
    predictions = bagged._hybrid_vote_on_score_agreement(
        mean_predictions=[0, 1, 2, 3],
        vote_predictions=[3, 2, 1, 0],
        score_mean_predictions=[0, 0, 2, 1],
    )

    assert predictions == [3, 1, 1, 3]


def test_bagged_gate_rescues_fresh_train_sample_stress(tmp_path) -> None:
    payload = bagged.build_gate(
        output_dir=tmp_path / "out",
        train_sample_seeds=(1729, 2027),
        split_seeds=(1729, 1731, 1733),
        bootstrap_samples=100,
        run_date="2026-05-01",
    )
    headline = payload["headline"]

    assert payload["pass_gate"] is True
    assert headline["component_model_count"] == 6
    assert headline["new_train_sample_seed_count"] == 1
    assert headline["selected_minus_best_label_copy"] >= 0.02
    assert headline["selected_minus_score_only_bagged_control"] >= 0.02
    assert headline["selected_minus_zero_hidden_control"] >= 0.02
    assert headline["paired_ci95_low_vs_best_label_copy"] > 0.0
    assert headline["paired_ci95_low_vs_score_only_bagged"] > 0.0
    assert headline["wrong_example_hidden_control_accuracy"] <= headline["best_label_copy_eval_accuracy"]
    assert headline["candidate_roll_hidden_control_accuracy"] <= headline["best_label_copy_eval_accuracy"]
    assert payload["jackknife_summary"]["row_count"] == 0
    assert payload["jackknife_summary"]["all_pass"] is True
    assert payload["packet_contract"]["raw_payload_bytes"] == 2
    assert payload["packet_contract"]["framed_record_bytes"] == 5
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["packet_contract"]["source_kv_exposed"] is False
    assert payload["packet_contract"]["raw_hidden_vector_transmitted"] is False
    assert payload["packet_contract"]["raw_scores_transmitted"] is False


def test_bagged_gate_requires_fresh_train_sample_for_promotion(tmp_path) -> None:
    payload = bagged.build_gate(
        output_dir=tmp_path / "out",
        train_sample_seeds=(1729,),
        split_seeds=(1729, 1731),
        bootstrap_samples=100,
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is False
    assert payload["headline"]["new_train_sample_seed_count"] == 0
    assert payload["headline"]["component_model_count"] == 2


def test_bagged_gate_reports_three_sample_jackknife(tmp_path) -> None:
    payload = bagged.build_gate(
        output_dir=tmp_path / "out",
        train_sample_seeds=(1729, 2027, 2039),
        split_seeds=(1729, 1731, 1733),
        bootstrap_samples=100,
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["component_model_count"] == 9
    assert payload["headline"]["new_train_sample_seed_count"] == 2
    assert payload["jackknife_summary"]["row_count"] == 3
    assert payload["jackknife_summary"]["pass_count"] == 3
    assert payload["jackknife_summary"]["selected_minus_best_label_copy_min"] >= 0.02
    assert payload["jackknife_summary"]["paired_ci95_low_vs_best_label_copy_min"] > 0.0
    assert payload["jackknife_summary"]["selected_minus_score_only_bagged_control_min"] >= 0.02
    assert payload["jackknife_summary"]["paired_ci95_low_vs_score_only_bagged_min"] > 0.0


def test_bagged_gate_writes_artifacts(tmp_path) -> None:
    bagged.build_gate(
        output_dir=tmp_path / "out",
        train_sample_seeds=(1729,),
        split_seeds=(1729, 1731),
        bootstrap_samples=100,
        run_date="2026-05-01",
    )

    summary = tmp_path / "out" / "hellaswag_hidden_innovation_bagged_gate.json"
    rows = tmp_path / "out" / "component_rows.csv"
    assert summary.exists()
    assert rows.exists()
    assert (tmp_path / "out" / "jackknife_rows.csv").exists()
    assert (tmp_path / "out" / "hellaswag_hidden_innovation_bagged_gate.md").exists()
    assert (tmp_path / "out" / "predictions.jsonl").exists()
    assert (tmp_path / "out" / "sample_caches.jsonl").exists()
    assert (tmp_path / "out" / "manifest.json").exists()

    parsed = json.loads(summary.read_text(encoding="utf-8"))
    assert parsed["gate"] == "source_private_hellaswag_hidden_innovation_bagged_gate"
    with rows.open(encoding="utf-8", newline="") as handle:
        parsed_rows = list(csv.DictReader(handle))
    assert len(parsed_rows) == 4
    assert {row["view"] for row in parsed_rows} == {"score_only", "score_hidden_residual"}
