from __future__ import annotations

import json
import random

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
    aggregate_row = payload["rows"][0]
    assert "learned_synonym_dictionary_packet" in metrics
    assert "atom_id_derangement" in metrics
    assert "top_atom_knockout" in metrics
    assert "private_random_knockout" in metrics
    assert row["budget_bytes"] == 4
    assert aggregate_row["best_control_name"] in metrics
    assert "top_atom_knockout_accuracy" in aggregate_row
    assert "private_random_knockout_accuracy" in aggregate_row
    assert "private_random_knockout_lift_reduction" in aggregate_row
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


def test_learned_synonym_dictionary_hashed_hf_feature_mode_records_model(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_hf_text_features(texts: list[str], *, dim: int, text_feature_mode: str) -> np.ndarray:
        calls.append(text_feature_mode)
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
        output_dir=tmp_path / "learned_synonym_dictionary_hashed_hf_features",
        budgets=[4],
        train_examples=16,
        eval_examples=8,
        seed=25,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=32,
        text_feature_mode="hashed_hf_last_mean",
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
            / "learned_synonym_dictionary_hashed_hf_features"
            / "learned_synonym_dictionary_packet_gate.json"
        ).read_text()
    )
    assert payload["text_feature_mode"] == "hashed_hf_last_mean"
    assert summary["feature_model"] == "fake/local-model"
    assert summary["feature_device"] == "cpu"
    assert "hf_last_mean" in calls


def test_public_adapter_semantic_anchor_teacher_records_mode_and_audit(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_hf_text_features(texts: list[str], *, dim: int, text_feature_mode: str) -> np.ndarray:
        calls.append(text_feature_mode)
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
        output_dir=tmp_path / "learned_synonym_dictionary_public_adapter",
        budgets=[4],
        train_examples=16,
        eval_examples=8,
        seed=27,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=32,
        text_feature_mode="hf_last_mean",
        adapter_target_mode="semantic_anchor_teacher",
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
            / "learned_synonym_dictionary_public_adapter"
            / "learned_synonym_dictionary_packet_gate.json"
        ).read_text()
    )
    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_public_adapter" / "core_to_holdout" / "summary.json").read_text()
    )
    assert payload["adapter_target_mode"] == "semantic_anchor_teacher"
    assert payload["decoder_score_mode"] == "global_dot"
    assert summary["adapter_target_mode"] == "semantic_anchor_teacher"
    assert summary["decoder_score_mode"] == "global_dot"
    assert direction["adapter_target_mode"] == "semantic_anchor_teacher"
    assert "private_random_source_atoms" in direction["source_destroying_controls"]
    assert "permuted_teacher_receiver" in direction["source_destroying_controls"]
    assert "permuted_teacher_receiver" in direction["conditions"]
    assert "calibration_eval_exact_id_overlap_count" in direction["surface_overlap_audit"]
    assert direction["surface_overlap_audit"]["calibration_eval_exact_id_overlap_count"] >= 0
    assert "hf_last_mean" in calls


def test_public_adapter_fit_does_not_touch_private_source_or_answers(monkeypatch) -> None:
    def fail_answer_index(example: gate.Example) -> int:
        raise AssertionError("public adapter fit should not inspect answer labels")

    def fail_source_private_atoms(text: str, *, mode: str = "matched") -> dict[str, float]:
        raise AssertionError("public adapter fit should not inspect private source logs")

    monkeypatch.setattr(gate, "_answer_index", fail_answer_index)
    monkeypatch.setattr(gate, "_source_private_atoms", fail_source_private_atoms)
    examples = gate.make_benchmark(examples=12, candidates=4, seed=28, family_set="all")

    dictionary = gate._fit_dictionary(
        examples=examples,
        feature_dim=32,
        ridge=0.5,
        calibration_atom_view="synonym_stress",
        top_k=6,
        min_score=0.0,
        text_feature_mode="hashed",
        adapter_target_mode="semantic_anchor_teacher",
        receiver_mode="atom_ridge",
        contrastive_negative_sources=0,
        contrastive_rank=4,
    )

    assert dictionary.receiver_mode == "atom_ridge"
    assert dictionary.weights.shape == (32, len(gate.ATOM_ORDER))


def test_permuted_public_adapter_target_is_deterministic_negative_control() -> None:
    text = "tie-breaking round toward larger value"
    teacher = gate._semantic_anchor_atom_vector(text)
    permuted_a = gate._permute_atom_vector(teacher, namespace="semantic-anchor-teacher-negative")
    permuted_b = gate._permute_atom_vector(teacher, namespace="semantic-anchor-teacher-negative")

    assert np.any(teacher)
    assert np.array_equal(permuted_a, permuted_b)
    assert not np.array_equal(teacher, permuted_a)


def test_private_random_source_atoms_preserve_values_but_destroy_atom_ids() -> None:
    atoms = {
        gate.ATOM_ORDER[0]: 0.9,
        gate.ATOM_ORDER[1]: 0.6,
        gate.ATOM_ORDER[2]: 0.3,
    }
    randomized = gate._private_random_source_atoms(atoms, rng=random.Random(7))

    assert len(randomized) == len(atoms)
    assert not (set(randomized) & set(atoms))
    assert sorted(randomized.values(), reverse=True) == sorted(atoms.values(), reverse=True)


def test_all_public_eval_disjoint_calibration_uses_qualified_ids() -> None:
    train_rows = gate.make_benchmark(examples=16, candidates=4, seed=31, family_set="core")
    eval_rows = gate.make_benchmark(examples=16, candidates=4, seed=32, family_set="holdout")
    calibration_rows = gate._calibration_examples(
        mode="all_public_eval_disjoint",
        train_examples=train_rows,
        eval_examples=eval_rows,
        calibration_count=32,
        seed=33,
    )
    audit = gate._surface_overlap_audit(
        calibration_rows=calibration_rows,
        eval_rows=eval_rows,
        calibration_atom_view="synonym_stress",
        candidate_atom_view="heldout_synonym",
    )

    assert audit["calibration_eval_exact_id_overlap_count"] == 0
    assert audit["calibration_eval_local_id_overlap_count"] > 0
    assert not (
        {gate._qualified_example_id(example) for example in calibration_rows}
        & {gate._qualified_example_id(example) for example in eval_rows}
    )


def test_candidate_local_residual_scoring_ignores_common_candidate_component() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=30, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row

    class DummyDictionary:
        receiver_mode = "atom_ridge"

        def __init__(self, common: np.ndarray) -> None:
            self.common = common

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text] + self.common

    payload_atoms = {gate.ATOM_ORDER[0]: 1.0}
    baseline_scores, baseline_meta = gate._score_candidates(
        example=example,
        payload_atoms=payload_atoms,
        dictionary=DummyDictionary(np.zeros(dim, dtype=np.float64)),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="candidate_local_residual_norm",
    )
    shifted_scores, shifted_meta = gate._score_candidates(
        example=example,
        payload_atoms=payload_atoms,
        dictionary=DummyDictionary(np.ones(dim, dtype=np.float64) * 3.0),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="candidate_local_residual_norm",
    )

    assert np.allclose(baseline_scores, shifted_scores)
    assert baseline_meta["decoder_score_mode"] == "candidate_local_residual_norm"
    assert shifted_meta["candidate_local_payload_l2"] == baseline_meta["candidate_local_payload_l2"]


def test_candidate_local_innovation_residual_centers_payload_on_candidate_pool() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=30, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row

    class DummyDictionary:
        receiver_mode = "atom_ridge"

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    local_mean_payload = {gate.ATOM_ORDER[idx]: 0.25 for idx in range(4)}
    centered_scores, centered_meta = gate._score_candidates(
        example=example,
        payload_atoms=local_mean_payload,
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="candidate_local_innovation_residual_norm",
    )
    answer_scores, answer_meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="candidate_local_innovation_residual_norm",
    )

    assert np.allclose(centered_scores, 0.0)
    assert centered_meta["candidate_local_payload_transform"] == "subtract_candidate_pool_mean"
    assert centered_meta["candidate_local_payload_l2"] == 0.0
    assert answer_meta["candidate_local_raw_payload_l2"] == 1.0
    assert answer_scores[0] == max(answer_scores)


def test_permuted_null_gap_decoder_subtracts_null_receiver_scores() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=31, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row

    class ActiveDictionary:
        receiver_mode = "atom_ridge"

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    class NullDictionary:
        receiver_mode = "atom_ridge"

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return np.zeros(dim, dtype=np.float64)

    payload = gate._encode_atoms({gate.ATOM_ORDER[0]: 1.0}, budget_bytes=4)
    prediction, meta = gate._predict_from_payload(
        example=example,
        payload=payload,
        budget_bytes=4,
        dictionary=ActiveDictionary(),  # type: ignore[arg-type]
        null_dictionary=NullDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="candidate_local_permuted_null_gap_residual_norm",
        min_decision_score=0.0,
        permuted_null_weight=0.75,
    )

    assert prediction == example.candidates[0].label
    assert meta["decoder_score_mode"] == "candidate_local_permuted_null_gap_residual_norm"
    assert meta["decoder_score_base_mode"] == "candidate_local_residual_norm"
    assert meta["permuted_null_weight"] == 0.75
    assert meta["permuted_null_scores"] == [0.0, 0.0, 0.0, 0.0]
    assert meta["scores"] == meta["active_scores"]


def test_candidate_local_random_rotation_sign_sketch_is_deterministic_and_local() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=31, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row

    class DummyDictionary:
        receiver_mode = "atom_ridge"

        def __init__(self, common: np.ndarray) -> None:
            self.common = common

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text] + self.common

    for mode, quantization in {
        "candidate_local_random_rotation_sign_residual_norm": "sign",
        "candidate_local_random_rotation_rank_sign_residual_norm": "rank_sign",
    }.items():
        kwargs = {
            "example": example,
            "payload_atoms": {gate.ATOM_ORDER[0]: 1.0},
            "candidate_atom_view": "native",
            "decoder_score_mode": mode,
        }
        baseline_scores, baseline_meta = gate._score_candidates(
            dictionary=DummyDictionary(np.zeros(dim, dtype=np.float64)),  # type: ignore[arg-type]
            **kwargs,
        )
        repeat_scores, repeat_meta = gate._score_candidates(
            dictionary=DummyDictionary(np.zeros(dim, dtype=np.float64)),  # type: ignore[arg-type]
            **kwargs,
        )
        shifted_scores, shifted_meta = gate._score_candidates(
            dictionary=DummyDictionary(np.ones(dim, dtype=np.float64) * 5.0),  # type: ignore[arg-type]
            **kwargs,
        )

        assert np.allclose(baseline_scores, repeat_scores)
        assert np.allclose(baseline_scores, shifted_scores)
        assert baseline_meta["decoder_score_mode"] == mode
        assert baseline_meta["candidate_local_transform"] == "public_orthogonal_sign_sketch"
        assert baseline_meta["candidate_local_quantization"] == quantization
        assert baseline_meta["candidate_local_sketch_bits"] == dim
        assert repeat_meta["candidate_local_transform_namespace"] == baseline_meta["candidate_local_transform_namespace"]
        assert shifted_meta["candidate_local_payload_l2"] == baseline_meta["candidate_local_payload_l2"]


def test_relative_anchor_dot_scores_in_anchor_coordinates() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=32, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row

    class DummyDictionary:
        receiver_mode = "atom_ridge"
        relative_anchor_vectors = np.eye(dim, dtype=np.float64)

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    scores, meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="relative_anchor_dot",
    )

    assert meta["decoder_score_mode"] == "relative_anchor_dot"
    assert meta["relative_anchor_count"] == dim
    assert scores[0] == max(scores)


def test_relative_anchor_innovation_residual_records_local_prior() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=33, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row

    class DummyDictionary:
        receiver_mode = "atom_ridge"
        relative_anchor_vectors = np.eye(dim, dtype=np.float64)

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    scores, meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="relative_anchor_innovation_residual_norm",
    )

    assert meta["decoder_score_mode"] == "relative_anchor_innovation_residual_norm"
    assert meta["relative_anchor_count"] == dim
    assert meta["relative_anchor_local_mean_l2"] > 0
    assert scores[0] == max(scores)


def test_rank_normalized_rows_preserves_anchor_order() -> None:
    ranked = gate._rank_normalized_rows(np.array([[0.2, 0.5, 0.2], [3.0, 1.0, 2.0]], dtype=np.float64))

    assert ranked.shape == (2, 3)
    assert ranked[0, 1] == 1.0
    assert ranked[0, 0] < ranked[0, 2]
    assert np.allclose(ranked[1], np.array([1.0, -1.0, 0.0]))
    assert np.allclose(gate._rank_normalized_rows(np.ones((2, 1), dtype=np.float64)), 0.0)


def test_relative_anchor_rank_innovation_records_rank_prior() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=33, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row

    class DummyDictionary:
        receiver_mode = "atom_ridge"
        relative_anchor_vectors = np.eye(dim, dtype=np.float64)

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    scores, meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="relative_anchor_rank_innovation_residual_norm",
    )

    assert len(scores) == len(example.candidates)
    assert meta["decoder_score_mode"] == "relative_anchor_rank_innovation_residual_norm"
    assert meta["relative_anchor_rank_normalized"] is True
    assert meta["relative_anchor_count"] == dim
    assert meta["relative_anchor_local_mean_l2"] > 0


def test_relative_anchor_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_relative_anchor",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=33,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="relative_anchor_dot",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_relative_anchor" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_relative_anchor" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["decoder_score_mode"] == "relative_anchor_dot"
    assert direction["decoder_score_mode"] == "relative_anchor_dot"
    assert prediction["metadata"]["decoder_score_mode"] == "relative_anchor_dot"
    assert prediction["metadata"]["relative_anchor_count"] > 0


def test_random_rotation_sign_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_random_rotation_sign",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=34,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="candidate_local_random_rotation_sign_residual_norm",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (
            tmp_path
            / "learned_synonym_dictionary_random_rotation_sign"
            / "core_to_holdout"
            / "summary.json"
        ).read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path
            / "learned_synonym_dictionary_random_rotation_sign"
            / "core_to_holdout"
            / "predictions_budget4.jsonl"
        )
        .read_text()
        .splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )

    assert payload["decoder_score_mode"] == "candidate_local_random_rotation_sign_residual_norm"
    assert direction["decoder_score_mode"] == "candidate_local_random_rotation_sign_residual_norm"
    assert prediction["metadata"]["decoder_score_mode"] == "candidate_local_random_rotation_sign_residual_norm"
    assert prediction["metadata"]["candidate_local_transform"] == "public_orthogonal_sign_sketch"


def test_permuted_null_gap_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_permuted_null_gap",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=35,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="train_only",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="candidate_local_permuted_null_gap_residual_norm",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
        permuted_null_weight=0.75,
    )

    direction = json.loads(
        (
            tmp_path
            / "learned_synonym_dictionary_permuted_null_gap"
            / "core_to_holdout"
            / "summary.json"
        ).read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path
            / "learned_synonym_dictionary_permuted_null_gap"
            / "core_to_holdout"
            / "predictions_budget4.jsonl"
        )
        .read_text()
        .splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )

    assert payload["decoder_score_mode"] == "candidate_local_permuted_null_gap_residual_norm"
    assert payload["permuted_null_weight"] == 0.75
    assert direction["decoder_score_mode"] == "candidate_local_permuted_null_gap_residual_norm"
    assert direction["permuted_null_weight"] == 0.75
    assert prediction["metadata"]["decoder_score_mode"] == "candidate_local_permuted_null_gap_residual_norm"
    assert prediction["metadata"]["decoder_score_base_mode"] == "candidate_local_residual_norm"
    assert "permuted_null_scores" in prediction["metadata"]


def test_procrustes_dot_maps_payload_with_public_rotation() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=34, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row
    procrustes_matrix = np.eye(dim, dtype=np.float64)
    procrustes_matrix[0, :] = 0.0
    procrustes_matrix[:, 1] = 0.0
    procrustes_matrix[0, 1] = 1.0

    class DummyDictionary:
        receiver_mode = "atom_ridge"

        def __init__(self, matrix: np.ndarray) -> None:
            self.procrustes_matrix = matrix

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    scores, meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(procrustes_matrix),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="procrustes_dot",
    )

    assert meta["decoder_score_mode"] == "procrustes_dot"
    assert meta["procrustes_payload_l2"] == 1.0
    assert scores[1] == max(scores)


def test_procrustes_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_procrustes",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=35,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="procrustes_dot",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_procrustes" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_procrustes" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["decoder_score_mode"] == "procrustes_dot"
    assert direction["decoder_score_mode"] == "procrustes_dot"
    assert prediction["metadata"]["decoder_score_mode"] == "procrustes_dot"
    assert prediction["metadata"]["procrustes_payload_l2"] > 0


def test_ridge_cca_dot_scores_in_canonical_coordinates() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=36, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row
    source_projection = np.zeros((dim, 1), dtype=np.float64)
    target_projection = np.zeros((dim, 1), dtype=np.float64)
    source_projection[0, 0] = 1.0
    target_projection[1, 0] = 1.0

    class DummyDictionary:
        receiver_mode = "atom_ridge"
        cca_rank = 1
        cca_source_mean = np.zeros(dim, dtype=np.float64)
        cca_target_mean = np.zeros(dim, dtype=np.float64)
        cca_source_projection = source_projection
        cca_target_projection = target_projection
        cca_correlations = np.ones(1, dtype=np.float64)

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    scores, meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="ridge_cca_dot",
    )

    assert meta["decoder_score_mode"] == "ridge_cca_dot"
    assert meta["cca_rank"] == 1
    assert meta["cca_payload_l2"] == 1.0
    assert scores[1] == max(scores)


def test_ridge_cca_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_ridge_cca",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=37,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="ridge_cca_dot",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_ridge_cca" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_ridge_cca" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["decoder_score_mode"] == "ridge_cca_dot"
    assert direction["decoder_score_mode"] == "ridge_cca_dot"
    assert prediction["metadata"]["decoder_score_mode"] == "ridge_cca_dot"
    assert prediction["metadata"]["cca_rank"] > 0


def test_ridge_cca_residual_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_ridge_cca_residual",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=38,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="ridge_cca_residual_norm",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_ridge_cca_residual" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path
            / "learned_synonym_dictionary_ridge_cca_residual"
            / "core_to_holdout"
            / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["decoder_score_mode"] == "ridge_cca_residual_norm"
    assert direction["decoder_score_mode"] == "ridge_cca_residual_norm"
    assert prediction["metadata"]["decoder_score_mode"] == "ridge_cca_residual_norm"
    assert prediction["metadata"]["cca_rank"] > 0


def test_lstirp_relative_dot_translates_between_relative_bases() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=39, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row
    translation = np.zeros((dim, dim), dtype=np.float64)
    translation[0, 1] = 1.0

    class DummyDictionary:
        receiver_mode = "atom_ridge"
        lstirp_source_anchor_vectors = np.eye(dim, dtype=np.float64)
        lstirp_target_anchor_vectors = np.eye(dim, dtype=np.float64)
        lstirp_source_relative_mean = np.zeros(dim, dtype=np.float64)
        lstirp_target_relative_mean = np.zeros(dim, dtype=np.float64)
        lstirp_translation = translation

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    scores, meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="lstirp_relative_dot",
    )

    assert meta["decoder_score_mode"] == "lstirp_relative_dot"
    assert meta["lstirp_source_anchor_count"] == dim
    assert meta["lstirp_target_anchor_count"] == dim
    assert scores[1] == max(scores)


def test_lstirp_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_lstirp",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=40,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="lstirp_relative_dot",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_lstirp" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_lstirp" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["decoder_score_mode"] == "lstirp_relative_dot"
    assert direction["decoder_score_mode"] == "lstirp_relative_dot"
    assert prediction["metadata"]["decoder_score_mode"] == "lstirp_relative_dot"
    assert prediction["metadata"]["lstirp_source_anchor_count"] > 0
    assert prediction["metadata"]["lstirp_target_anchor_count"] > 0


def test_lstirp_residual_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_lstirp_residual",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=41,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="lstirp_relative_residual_norm",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_lstirp_residual" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_lstirp_residual" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["decoder_score_mode"] == "lstirp_relative_residual_norm"
    assert direction["decoder_score_mode"] == "lstirp_relative_residual_norm"
    assert prediction["metadata"]["decoder_score_mode"] == "lstirp_relative_residual_norm"
    assert prediction["metadata"]["lstirp_source_anchor_count"] > 0


def test_inverse_relative_dot_maps_payload_through_inverse_projection() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=42, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row
    inverse_map = np.zeros((dim, dim), dtype=np.float64)
    inverse_map[0, 1] = 1.0

    class DummyDictionary:
        receiver_mode = "atom_ridge"
        inverse_relative_source_mean = np.zeros(dim, dtype=np.float64)
        inverse_relative_target_mean = np.zeros(dim, dtype=np.float64)
        inverse_relative_map = inverse_map
        inverse_relative_anchor_count = dim
        inverse_relative_condition_number = 1.0

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    scores, meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="inverse_relative_dot",
    )

    assert meta["decoder_score_mode"] == "inverse_relative_dot"
    assert meta["inverse_relative_anchor_count"] == dim
    assert scores[1] == max(scores)


def test_inverse_relative_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_inverse_relative",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=43,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="inverse_relative_dot",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_inverse_relative" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_inverse_relative" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["decoder_score_mode"] == "inverse_relative_dot"
    assert direction["decoder_score_mode"] == "inverse_relative_dot"
    assert prediction["metadata"]["decoder_score_mode"] == "inverse_relative_dot"
    assert prediction["metadata"]["inverse_relative_anchor_count"] > 0


def test_ot_gw_dot_maps_payload_through_transport_plan() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=44, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row
    transport = np.zeros((dim, dim), dtype=np.float64)
    transport[0, 1] = 1.0

    class DummyDictionary:
        receiver_mode = "atom_ridge"
        ot_gw_transport_map = transport
        ot_gw_iterations = 3
        ot_gw_entropy = 0.08
        ot_gw_fused_weight = 0.35
        ot_gw_coupling_l1 = 1.0
        ot_gw_objective = 0.5

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    scores, meta = gate._score_candidates(
        example=example,
        payload_atoms={gate.ATOM_ORDER[0]: 1.0},
        dictionary=DummyDictionary(),  # type: ignore[arg-type]
        candidate_atom_view="native",
        decoder_score_mode="ot_gw_dot",
    )

    assert meta["decoder_score_mode"] == "ot_gw_dot"
    assert meta["ot_gw_iterations"] == 3
    assert meta["ot_gw_payload_l2"] == 1.0
    assert scores[1] == max(scores)


def test_explicit_transport_modes_map_payload_through_transport_plan() -> None:
    example = gate.make_benchmark(examples=1, candidates=4, seed=45, family_set="core")[0]
    dim = len(gate.ATOM_ORDER)
    rows = {}
    for idx, candidate in enumerate(example.candidates):
        row = np.zeros(dim, dtype=np.float64)
        row[idx] = 1.0
        rows[candidate.patch_intent] = row
    transport = np.zeros((dim, dim), dtype=np.float64)
    transport[0, 1] = 1.0

    class DummyDictionary:
        receiver_mode = "atom_ridge"
        sinkhorn_ot_transport_map = transport
        sinkhorn_ot_entropy = 0.08
        sinkhorn_ot_coupling_l1 = 1.0
        sinkhorn_ot_objective = 0.4
        ot_gw_transport_map = transport
        ot_gw_iterations = 3
        ot_gw_sinkhorn_iterations = 80
        ot_gw_entropy = 0.08
        ot_gw_fused_weight = 0.35
        ot_gw_coupling_l1 = 1.0
        ot_gw_objective = 0.5

        def predict_vector(self, text: str, *, apply_top_k: bool = True) -> np.ndarray:
            return rows[text]

    for mode, prefix, kind in (
        ("sinkhorn_ot_dot", "sinkhorn_ot", "sinkhorn_ot"),
        ("gromov_wasserstein_dot", "ot_gw", "fused_gromov_wasserstein"),
    ):
        scores, meta = gate._score_candidates(
            example=example,
            payload_atoms={gate.ATOM_ORDER[0]: 1.0},
            dictionary=DummyDictionary(),  # type: ignore[arg-type]
            candidate_atom_view="native",
            decoder_score_mode=mode,
        )

        assert scores[1] == max(scores)
        assert meta["decoder_score_mode"] == mode
        assert meta["transport_kind"] == kind
        assert meta[f"{prefix}_payload_l2"] == 1.0


def test_ot_gw_receiver_records_mode(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_ot_gw",
        budgets=[4],
        train_examples=16,
        eval_examples=6,
        seed=45,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=16,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        adapter_target_mode="semantic_anchor_teacher",
        decoder_score_mode="ot_gw_dot",
        receiver_mode="atom_ridge",
        ridge=0.5,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_ot_gw" / "core_to_holdout" / "summary.json").read_text()
    )
    prediction = next(
        json.loads(line)
        for line in (
            tmp_path / "learned_synonym_dictionary_ot_gw" / "core_to_holdout" / "predictions_budget4.jsonl"
        ).read_text().splitlines()
        if json.loads(line)["condition"] == "learned_synonym_dictionary_packet"
    )
    assert payload["decoder_score_mode"] == "ot_gw_dot"
    assert direction["decoder_score_mode"] == "ot_gw_dot"
    assert prediction["metadata"]["decoder_score_mode"] == "ot_gw_dot"
    assert prediction["metadata"]["ot_gw_iterations"] > 0


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


def test_low_rank_factor_receiver_scores_with_explicit_factors(monkeypatch) -> None:
    def fake_featurize_text(text: str, *, dim: int, text_feature_mode: str = "hashed") -> np.ndarray:
        assert text == "candidate"
        assert dim == 3
        assert text_feature_mode == "hashed"
        return np.array([1.0, 2.0, -1.0], dtype=np.float64)

    monkeypatch.setattr(gate, "_featurize_text", fake_featurize_text)
    left = np.array([[1.0, 0.0], [0.5, -1.0], [0.0, 2.0]], dtype=np.float64)
    right = np.zeros((len(gate.ATOM_ORDER), 2), dtype=np.float64)
    right[gate.ATOM_TO_ID["sum"], :] = [2.0, -0.5]
    right[gate.ATOM_TO_ID["default"], :] = [1.0, 1.5]
    dictionary = gate.LearnedSynonymDictionary(
        feature_dim=3,
        weights=left @ right.T,
        top_k=2,
        min_score=0.0,
        text_feature_mode="hashed",
        receiver_mode="contrastive_low_rank_factor",
        bias=0.25,
        contrastive_rank=2,
        receiver_effective_rank=2,
        left_factors=left,
        right_factors=right,
    )

    payload_atoms = {"sum": 0.75, "default": 0.25}
    feature = np.array([1.0, 2.0, -1.0], dtype=np.float64)
    atom_vector = gate._atom_vector(payload_atoms)
    expected = float((feature @ left) @ (atom_vector @ right) + 0.25)
    assert dictionary.score_text("candidate", payload_atoms) == expected


def test_direct_low_rank_factor_receiver_records_training_args(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_direct_low_rank",
        budgets=[4],
        train_examples=20,
        eval_examples=8,
        seed=41,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=20,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        receiver_mode="contrastive_low_rank_factor",
        contrastive_negative_sources=1,
        contrastive_rank=3,
        low_rank_factor_epochs=8,
        low_rank_factor_lr=0.03,
        low_rank_factor_loss="squared",
        low_rank_factor_seed=123,
        ridge=0.001,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_direct_low_rank" / "core_to_holdout" / "summary.json").read_text()
    )
    assert payload["receiver_mode"] == "contrastive_low_rank_factor"
    assert payload["contrastive_rank"] == 3
    assert payload["low_rank_factor_epochs"] == 8
    assert payload["low_rank_factor_lr"] == 0.03
    assert payload["low_rank_factor_loss"] == "squared"
    assert payload["low_rank_factor_seed"] == 123
    assert direction["receiver_mode"] == "contrastive_low_rank_factor"
    assert direction["contrastive_rank"] == 3
    assert direction["low_rank_factor_epochs"] == 8
    assert direction["low_rank_factor_lr"] == 0.03
    assert direction["low_rank_factor_loss"] == "squared"
    assert direction["low_rank_factor_seed"] == 123
    assert direction["receiver_effective_rank"] <= 3


def test_fit_low_rank_factor_receiver_is_deterministic() -> None:
    examples = gate.make_benchmark(examples=18, candidates=4, seed=43, family_set="all")
    kwargs = dict(
        examples=examples,
        feature_dim=36,
        ridge=0.001,
        calibration_atom_view="synonym_stress",
        top_k=6,
        min_score=0.0,
        text_feature_mode="semantic_anchor",
        receiver_mode="contrastive_low_rank_factor",
        contrastive_negative_sources=1,
        contrastive_rank=2,
        low_rank_factor_epochs=6,
        low_rank_factor_lr=0.04,
        low_rank_factor_loss="bce",
        low_rank_factor_seed=777,
        seed=43,
    )

    first = gate._fit_dictionary(**kwargs)
    second = gate._fit_dictionary(**kwargs)

    assert first.receiver_mode == "contrastive_low_rank_factor"
    assert first.left_factors is not None
    assert first.right_factors is not None
    assert first.left_factors.shape == (36, 2)
    assert first.right_factors.shape == (len(gate.ATOM_ORDER), 2)
    assert first.receiver_effective_rank <= 2
    assert np.allclose(first.left_factors, second.left_factors)
    assert np.allclose(first.right_factors, second.right_factors)
    assert first.bias == second.bias


def test_jepa_query_resampler_scores_with_attention_factors(monkeypatch) -> None:
    def fake_featurize_text(text: str, *, dim: int, text_feature_mode: str = "hashed") -> np.ndarray:
        assert text == "candidate"
        assert dim == 2
        assert text_feature_mode == "hashed"
        return np.array([1.0, 0.5], dtype=np.float64)

    monkeypatch.setattr(gate, "_featurize_text", fake_featurize_text)
    query_factors = np.zeros((2, 2, 2), dtype=np.float64)
    query_factors[0, 0, :] = [1.0, 0.0]
    query_factors[1, 0, :] = [0.0, 1.0]
    query_factors[0, 1, :] = [0.5, 0.5]
    query_factors[1, 1, :] = [1.0, -0.5]
    atom_keys = np.zeros((len(gate.ATOM_ORDER), 2), dtype=np.float64)
    atom_values = np.zeros((len(gate.ATOM_ORDER), 2), dtype=np.float64)
    atom_keys[gate.ATOM_TO_ID["sum"], :] = [1.0, 0.0]
    atom_keys[gate.ATOM_TO_ID["default"], :] = [0.0, 1.0]
    atom_values[gate.ATOM_TO_ID["sum"], :] = [0.25, 1.0]
    atom_values[gate.ATOM_TO_ID["default"], :] = [1.0, -0.25]
    output = np.array([0.5, -0.25, 0.75, 0.1], dtype=np.float64)
    dictionary = gate.LearnedSynonymDictionary(
        feature_dim=2,
        weights=np.zeros((2, len(gate.ATOM_ORDER)), dtype=np.float64),
        top_k=2,
        min_score=0.0,
        text_feature_mode="hashed",
        receiver_mode="jepa_query_resampler",
        bias=0.125,
        receiver_effective_rank=2,
        resampler_query_factors=query_factors,
        resampler_atom_keys=atom_keys,
        resampler_atom_values=atom_values,
        resampler_output=output,
        jepa_query_count=2,
        jepa_hidden_dim=2,
    )

    feature = np.array([1.0, 0.5], dtype=np.float64)
    atom_vector = gate._atom_vector({"sum": 1.0, "default": 0.5})
    context, _ = gate._jepa_attention_features(
        features=feature,
        payload_vector=atom_vector,
        query_factors=query_factors,
        atom_keys=atom_keys,
        atom_values=atom_values,
    )
    expected = float(context @ output + 0.125)
    assert dictionary.score_text("candidate", {"sum": 1.0, "default": 0.5}) == expected


def test_jepa_query_resampler_receiver_records_bottleneck(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_jepa_query",
        budgets=[4],
        train_examples=20,
        eval_examples=8,
        seed=47,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=20,
        feature_dim=40,
        text_feature_mode="semantic_anchor",
        receiver_mode="jepa_query_resampler",
        contrastive_negative_sources=1,
        jepa_query_count=3,
        jepa_hidden_dim=5,
        ridge=0.01,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (tmp_path / "learned_synonym_dictionary_jepa_query" / "core_to_holdout" / "summary.json").read_text()
    )
    assert payload["receiver_mode"] == "jepa_query_resampler"
    assert payload["jepa_query_count"] == 3
    assert payload["jepa_hidden_dim"] == 5
    assert direction["receiver_mode"] == "jepa_query_resampler"
    assert direction["jepa_query_count"] == 3
    assert direction["jepa_hidden_dim"] == 5
    assert direction["jepa_query_entropy"] is not None
    assert direction["jepa_context_variance"] is not None
    assert direction["receiver_effective_rank"] is not None


def test_fit_jepa_query_resampler_receiver_is_deterministic() -> None:
    examples = gate.make_benchmark(examples=18, candidates=4, seed=49, family_set="all")
    kwargs = dict(
        examples=examples,
        feature_dim=36,
        ridge=0.01,
        calibration_atom_view="synonym_stress",
        top_k=6,
        min_score=0.0,
        text_feature_mode="semantic_anchor",
        receiver_mode="jepa_query_resampler",
        contrastive_negative_sources=1,
        contrastive_rank=2,
        jepa_query_count=4,
        jepa_hidden_dim=6,
        seed=49,
    )

    first = gate._fit_dictionary(**kwargs)
    second = gate._fit_dictionary(**kwargs)

    assert first.receiver_mode == "jepa_query_resampler"
    assert first.resampler_query_factors is not None
    assert first.resampler_atom_keys is not None
    assert first.resampler_atom_values is not None
    assert first.resampler_output is not None
    assert first.resampler_query_factors.shape == (36, 4, 6)
    assert first.resampler_atom_keys.shape == (len(gate.ATOM_ORDER), 6)
    assert first.resampler_atom_values.shape == (len(gate.ATOM_ORDER), 6)
    assert first.resampler_output.shape == (24,)
    assert np.allclose(first.resampler_query_factors, second.resampler_query_factors)
    assert np.allclose(first.resampler_atom_keys, second.resampler_atom_keys)
    assert np.allclose(first.resampler_atom_values, second.resampler_atom_values)
    assert np.allclose(first.resampler_output, second.resampler_output)
    assert first.bias == second.bias


def test_jepa_attention_empty_payload_returns_zero_context() -> None:
    query_factors = np.ones((3, 2, 4), dtype=np.float64)
    atom_keys = np.ones((len(gate.ATOM_ORDER), 4), dtype=np.float64)
    atom_values = np.ones((len(gate.ATOM_ORDER), 4), dtype=np.float64)
    context, attention = gate._jepa_attention_features(
        features=np.array([1.0, 0.0, -1.0], dtype=np.float64),
        payload_vector=np.zeros(len(gate.ATOM_ORDER), dtype=np.float64),
        query_factors=query_factors,
        atom_keys=atom_keys,
        atom_values=atom_values,
    )

    assert context.shape == (8,)
    assert attention.shape == (2, 0)
    assert np.allclose(context, 0.0)
    assert not np.isnan(context).any()


def test_trainable_jepa_query_resampler_records_training_metadata(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_trainable_jepa_query",
        budgets=[4],
        train_examples=12,
        eval_examples=6,
        seed=51,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=12,
        feature_dim=24,
        text_feature_mode="semantic_anchor",
        receiver_mode="jepa_query_resampler_trainable",
        contrastive_negative_sources=1,
        jepa_query_count=2,
        jepa_hidden_dim=4,
        jepa_train_epochs=2,
        jepa_lr=0.02,
        jepa_weight_decay=0.001,
        ridge=0.01,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (
            tmp_path
            / "learned_synonym_dictionary_trainable_jepa_query"
            / "core_to_holdout"
            / "summary.json"
        ).read_text()
    )
    row = direction["budget_summaries"][0]
    assert payload["receiver_mode"] == "jepa_query_resampler_trainable"
    assert payload["jepa_trainable_factors"] is True
    assert payload["jepa_train_epochs"] == 2
    assert payload["jepa_lr"] == 0.02
    assert payload["jepa_weight_decay"] == 0.001
    assert direction["jepa_trainable_factors"] is True
    assert direction["jepa_train_epochs"] == 2
    assert direction["jepa_lr"] == 0.02
    assert direction["jepa_weight_decay"] == 0.001
    assert "zero_source" in row["metrics"]
    assert "shuffled_source" in row["metrics"]
    assert "atom_id_derangement" in row["metrics"]
    assert "top_atom_knockout" in row["metrics"]
    assert "private_random_knockout" in row["metrics"]


def test_fit_trainable_jepa_query_resampler_is_deterministic() -> None:
    examples = gate.make_benchmark(examples=10, candidates=4, seed=53, family_set="all")
    kwargs = dict(
        examples=examples,
        feature_dim=20,
        ridge=0.01,
        calibration_atom_view="synonym_stress",
        top_k=6,
        min_score=0.0,
        text_feature_mode="semantic_anchor",
        receiver_mode="jepa_query_resampler_trainable",
        contrastive_negative_sources=1,
        contrastive_rank=2,
        jepa_query_count=2,
        jepa_hidden_dim=4,
        jepa_train_epochs=2,
        jepa_lr=0.02,
        jepa_weight_decay=0.001,
        seed=53,
    )

    first = gate._fit_dictionary(**kwargs)
    second = gate._fit_dictionary(**kwargs)

    assert first.receiver_mode == "jepa_query_resampler_trainable"
    assert first.jepa_trainable_factors is True
    assert first.resampler_query_factors is not None
    assert first.resampler_atom_keys is not None
    assert first.resampler_atom_values is not None
    assert first.resampler_output is not None
    assert np.allclose(first.resampler_query_factors, second.resampler_query_factors)
    assert np.allclose(first.resampler_atom_keys, second.resampler_atom_keys)
    assert np.allclose(first.resampler_atom_values, second.resampler_atom_values)
    assert np.allclose(first.resampler_output, second.resampler_output)
    assert first.bias == second.bias
    random_only = gate._fit_dictionary(
        **{k: v for k, v in kwargs.items() if k not in {"receiver_mode", "jepa_train_epochs", "jepa_lr", "jepa_weight_decay"}},
        receiver_mode="jepa_query_resampler",
    )
    assert not np.allclose(first.resampler_output, random_only.resampler_output)


def test_control_regularized_jepa_query_resampler_records_training_metadata(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_control_regularized_jepa_query",
        budgets=[4],
        train_examples=12,
        eval_examples=6,
        seed=55,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=12,
        feature_dim=24,
        text_feature_mode="semantic_anchor",
        receiver_mode="jepa_query_resampler_control_regularized",
        contrastive_negative_sources=1,
        jepa_query_count=2,
        jepa_hidden_dim=4,
        jepa_train_epochs=2,
        jepa_lr=0.02,
        jepa_weight_decay=0.001,
        ridge=0.01,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (
            tmp_path
            / "learned_synonym_dictionary_control_regularized_jepa_query"
            / "core_to_holdout"
            / "summary.json"
        ).read_text()
    )
    row = direction["budget_summaries"][0]
    assert payload["receiver_mode"] == "jepa_query_resampler_control_regularized"
    assert payload["jepa_trainable_factors"] is True
    assert payload["jepa_train_epochs"] == 2
    assert direction["receiver_mode"] == "jepa_query_resampler_control_regularized"
    assert direction["jepa_trainable_factors"] is True
    assert direction["jepa_train_epochs"] == 2
    assert "random_same_byte" in row["metrics"]
    assert "atom_id_derangement" in row["metrics"]


def test_fit_control_regularized_jepa_query_resampler_differs_from_plain_trainable() -> None:
    examples = gate.make_benchmark(examples=10, candidates=4, seed=57, family_set="all")
    kwargs = dict(
        examples=examples,
        feature_dim=20,
        ridge=0.01,
        calibration_atom_view="synonym_stress",
        top_k=6,
        min_score=0.0,
        text_feature_mode="semantic_anchor",
        contrastive_negative_sources=1,
        contrastive_rank=2,
        jepa_query_count=2,
        jepa_hidden_dim=4,
        jepa_train_epochs=2,
        jepa_lr=0.02,
        jepa_weight_decay=0.001,
        seed=57,
    )

    regularized = gate._fit_dictionary(
        **kwargs,
        receiver_mode="jepa_query_resampler_control_regularized",
    )
    plain = gate._fit_dictionary(
        **kwargs,
        receiver_mode="jepa_query_resampler_trainable",
    )
    repeated = gate._fit_dictionary(
        **kwargs,
        receiver_mode="jepa_query_resampler_control_regularized",
    )

    assert regularized.receiver_mode == "jepa_query_resampler_control_regularized"
    assert regularized.jepa_trainable_factors is True
    assert regularized.resampler_output is not None
    assert repeated.resampler_output is not None
    assert plain.resampler_output is not None
    assert np.allclose(regularized.resampler_output, repeated.resampler_output)
    assert not np.allclose(regularized.resampler_output, plain.resampler_output)


def test_pool_contrastive_jepa_query_resampler_records_training_metadata(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "learned_synonym_dictionary_pool_contrastive_jepa_query",
        budgets=[4],
        train_examples=12,
        eval_examples=6,
        seed=59,
        candidate_atom_view="heldout_synonym",
        calibration_atom_view="synonym_stress",
        candidate_calibration="all_public",
        calibration_examples=12,
        feature_dim=24,
        text_feature_mode="semantic_anchor",
        receiver_mode="jepa_query_resampler_pool_contrastive",
        contrastive_negative_sources=1,
        jepa_query_count=2,
        jepa_hidden_dim=4,
        jepa_train_epochs=2,
        jepa_lr=0.02,
        jepa_weight_decay=0.001,
        ridge=0.01,
        top_k=6,
        min_score=0.0,
        min_decision_score=0.3,
    )

    direction = json.loads(
        (
            tmp_path
            / "learned_synonym_dictionary_pool_contrastive_jepa_query"
            / "core_to_holdout"
            / "summary.json"
        ).read_text()
    )
    row = direction["budget_summaries"][0]
    assert payload["receiver_mode"] == "jepa_query_resampler_pool_contrastive"
    assert payload["jepa_trainable_factors"] is True
    assert payload["jepa_train_epochs"] == 2
    assert direction["receiver_mode"] == "jepa_query_resampler_pool_contrastive"
    assert direction["jepa_trainable_factors"] is True
    assert direction["jepa_train_epochs"] == 2
    assert "shuffled_source" in row["metrics"]
    assert "random_same_byte" in row["metrics"]
    assert "atom_id_derangement" in row["metrics"]


def test_fit_pool_contrastive_jepa_query_resampler_is_deterministic() -> None:
    examples = gate.make_benchmark(examples=10, candidates=4, seed=61, family_set="all")
    kwargs = dict(
        examples=examples,
        feature_dim=20,
        ridge=0.01,
        calibration_atom_view="synonym_stress",
        top_k=6,
        min_score=0.0,
        text_feature_mode="semantic_anchor",
        contrastive_negative_sources=1,
        contrastive_rank=2,
        jepa_query_count=2,
        jepa_hidden_dim=4,
        jepa_train_epochs=2,
        jepa_lr=0.02,
        jepa_weight_decay=0.001,
        seed=61,
    )

    first = gate._fit_dictionary(
        **kwargs,
        receiver_mode="jepa_query_resampler_pool_contrastive",
    )
    second = gate._fit_dictionary(
        **kwargs,
        receiver_mode="jepa_query_resampler_pool_contrastive",
    )
    control_regularized = gate._fit_dictionary(
        **kwargs,
        receiver_mode="jepa_query_resampler_control_regularized",
    )

    assert first.receiver_mode == "jepa_query_resampler_pool_contrastive"
    assert first.jepa_trainable_factors is True
    assert first.resampler_output is not None
    assert second.resampler_output is not None
    assert control_regularized.resampler_output is not None
    assert np.allclose(first.resampler_output, second.resampler_output)
    assert not np.allclose(first.resampler_output, control_regularized.resampler_output)
