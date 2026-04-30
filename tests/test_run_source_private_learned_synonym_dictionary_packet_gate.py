from __future__ import annotations

import json

import numpy as np

import scripts.run_source_private_learned_synonym_dictionary_packet_gate as gate
from scripts.run_source_private_learned_synonym_dictionary_packet_gate import run_gate


def test_learned_synonym_dictionary_gate_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary",
        budgets=[4],
        train_examples=32,
        eval_examples=12,
        seed=11,
        candidate_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=48,
        feature_dim=64,
        ridge=0.5,
        top_k=6,
        min_score=0.01,
    )

    summary = json.loads(
        (tmp_path / "learned_synonym_dictionary" / "learned_synonym_dictionary_packet_gate.json").read_text()
    )
    manifest = json.loads((tmp_path / "learned_synonym_dictionary" / "manifest.json").read_text())

    assert payload["gate"] == "source_private_learned_synonym_dictionary_packet_gate"
    assert summary["candidate_atom_view"] == "synonym_stress"
    assert set(summary["headline"]["direction_pass"]) == {"core_to_holdout", "holdout_to_core", "same_family_all"}
    assert "learned_synonym_dictionary_packet_gate.json" in manifest["artifacts"]
    assert (tmp_path / "learned_synonym_dictionary" / "core_to_holdout" / "predictions_budget4.jsonl").exists()


def test_learned_synonym_dictionary_rows_include_knockout_and_controls(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_controls",
        budgets=[4],
        train_examples=24,
        eval_examples=8,
        seed=13,
        candidate_atom_view="synonym_stress",
        candidate_calibration="train_only",
        calibration_examples=24,
        feature_dim=48,
        ridge=0.5,
        top_k=6,
        min_score=0.0,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_controls" / "core_to_holdout" / "summary.json").read_text()
    )
    row = direction["budget_summaries"][0]
    metrics = row["metrics"]
    assert "learned_synonym_dictionary_packet" in metrics
    assert "atom_id_derangement" in metrics
    assert "top_atom_knockout" in metrics
    assert "private_random_knockout" in metrics
    assert row["budget_bytes"] == 4
    assert row["paired_bootstrap_vs_target"]["ci95_high"] >= row["paired_bootstrap_vs_target"]["ci95_low"]
    assert payload["headline"]["max_learned_synonym_dictionary_accuracy"] >= 0.0


def test_learned_synonym_dictionary_supports_heldout_synonym_surface(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_heldout",
        budgets=[4],
        train_examples=24,
        eval_examples=8,
        seed=17,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=32,
        feature_dim=48,
        ridge=0.5,
        top_k=6,
        min_score=0.0,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_heldout" / "core_to_holdout" / "summary.json").read_text()
    )
    audit = direction["surface_overlap_audit"]
    assert payload["candidate_atom_view"] == "heldout_synonym"
    assert payload["calibration_atom_view"] == "synonym_stress"
    assert audit["candidate_atom_view"] == "heldout_synonym"
    assert audit["calibration_atom_view"] == "synonym_stress"
    assert audit["transformed_eval_surface_count"] > 0
    assert audit["exact_transformed_eval_surface_overlap_count"] == 0


def test_learned_synonym_dictionary_semantic_anchor_mode_records_threshold(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_semantic_anchor",
        budgets=[4],
        train_examples=24,
        eval_examples=8,
        seed=19,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=32,
        feature_dim=48,
        text_feature_mode="semantic_anchor",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.7,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_semantic_anchor" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_semantic_anchor" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "oracle_learned_candidate_atoms"
    )
    assert payload["text_feature_mode"] == "semantic_anchor"
    assert payload["min_decision_score"] == 0.7
    assert direction["text_feature_mode"] == "semantic_anchor"
    assert direction["min_decision_score"] == 0.7
    assert prediction["metadata"]["min_decision_score"] == 0.0


def test_learned_synonym_dictionary_hf_feature_mode_records_model(tmp_path, monkeypatch) -> None:
    def fake_hf_text_features(texts: list[str], *, dim: int, text_feature_mode: str) -> np.ndarray:
        rows = []
        for text in texts:
            row = np.zeros(dim, dtype=np.float64)
            for token in text.lower().split():
                row[sum(ord(ch) for ch in token) % dim] += 1.0
            norm = np.linalg.norm(row)
            rows.append(row / max(norm, 1.0))
        return np.stack(rows, axis=0)

    monkeypatch.setattr(gate, "_hf_text_features", fake_hf_text_features)
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_hf_features",
        budgets=[4],
        train_examples=16,
        eval_examples=8,
        seed=23,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=32,
        text_feature_mode="hf_mid_last_mean",
        feature_model="fake/local-model",
        feature_device="cpu",
        feature_dtype="float32",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.7,
    )

    summary = json.loads(
        (
            tmp_path
            / "learned_synonym_dictionary_hf_features"
            / "learned_synonym_dictionary_packet_gate.json"
        ).read_text()
    )
    assert payload["text_feature_mode"] == "hf_mid_last_mean"
    assert summary["feature_model"] == "fake/local-model"
    assert summary["feature_device"] == "cpu"
    assert summary["feature_dtype"] == "float32"


def test_learned_synonym_dictionary_contrastive_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_contrastive",
        budgets=[4],
        train_examples=24,
        eval_examples=8,
        seed=29,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=24,
        feature_dim=48,
        text_feature_mode="semantic_anchor",
        receiver_mode="contrastive_bilinear",
        contrastive_negative_sources=1,
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_contrastive" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_contrastive" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["receiver_mode"] == "contrastive_bilinear"
    assert payload["contrastive_negative_sources"] == 1
    assert direction["receiver_mode"] == "contrastive_bilinear"
    assert direction["contrastive_negative_sources"] == 1
    assert prediction["metadata"]["decoder"] in {
        "learned_synonym_dictionary",
        "learned_synonym_dictionary_target_preserve",
    }


def test_learned_synonym_dictionary_low_rank_query_receiver_records_bottleneck(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_low_rank_query",
        budgets=[4],
        train_examples=24,
        eval_examples=8,
        seed=31,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=24,
        feature_dim=48,
        text_feature_mode="semantic_anchor",
        receiver_mode="contrastive_low_rank_query",
        contrastive_negative_sources=1,
        contrastive_rank=3,
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_low_rank_query" / "core_to_holdout" / "summary.json").read_text()
    )
    assert payload["receiver_mode"] == "contrastive_low_rank_query"
    assert payload["contrastive_rank"] == 3
    assert direction["receiver_mode"] == "contrastive_low_rank_query"
    assert direction["contrastive_rank"] == 3
    assert direction["receiver_effective_rank"] <= 3


def test_low_rank_query_receiver_truncates_bilinear_map() -> None:
    examples = gate.make_benchmark(examples=18, candidates=4, seed=37, family_set="all")

    dictionary = gate._fit_dictionary(
        examples=examples,
        feature_dim=40,
        ridge=0.5,
        calibration_atom_view="synonym_stress",
        top_k=6,
        min_score=0.0,
        text_feature_mode="semantic_anchor",
        receiver_mode="contrastive_low_rank_query",
        contrastive_negative_sources=1,
        contrastive_rank=2,
        seed=37,
    )

    assert dictionary.receiver_mode == "contrastive_low_rank_query"
    assert dictionary.contrastive_rank == 2
    assert dictionary.receiver_effective_rank <= 2
    assert np.linalg.matrix_rank(dictionary.weights) <= 2
