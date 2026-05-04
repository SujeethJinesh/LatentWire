from __future__ import annotations

import json
import math

import numpy as np

from scripts import build_target_self_resonance_hellaswag_source_hidden_residual_slot_gate as gate


class _Row:
    def __init__(self, row_id: str) -> None:
        self.row_id = row_id


def test_hidden_source_features_are_roll_sensitive() -> None:
    hidden = np.eye(4, 6, dtype=np.float64)
    scores = np.asarray([0.1, 1.5, -0.2, 0.9], dtype=np.float64)

    features = gate._hidden_source_features(hidden=hidden, scores=scores, feature_mode="top2_delta")
    rolled = gate._candidate_roll_hidden_features(hidden=hidden, scores=scores, feature_mode="top2_delta")

    assert features.shape == (4 * 6 + 10,)
    assert features.dtype == np.float32
    assert not np.allclose(features, rolled)
    assert math.isclose(float(features[-10:-6].sum()), 1.0)
    assert math.isclose(float(features[-6:-2].sum()), 1.0)


def test_mean_top1_delta_feature_mode_is_smaller() -> None:
    hidden = np.arange(24, dtype=np.float64).reshape(4, 6)
    scores = [0.1, 0.2, 3.0, 0.4]

    features = gate._hidden_source_features(hidden=hidden, scores=scores, feature_mode="mean_top1_delta")

    assert features.shape == (2 * 6 + 10,)
    assert np.isfinite(features).all()


def test_target_score_derived_features_only_fill_score_tail() -> None:
    features = gate._target_score_derived_features(
        frozen_scores=[0.0, 2.0, 1.0, -1.0],
        feature_dim=30,
        score_feature_dim=10,
    )

    assert features.shape == (30,)
    assert np.allclose(features[:20], 0.0)
    assert not np.allclose(features[-10:], 0.0)


def test_source_top1_or_top2_oracle_uses_answer_when_in_top2() -> None:
    scores = [0.1, 3.0, 2.0, -1.0]

    in_top2 = gate._source_top1_or_top2_oracle_scores(scores, answer_index=2)
    outside_top2 = gate._source_top1_or_top2_oracle_scores(scores, answer_index=3)

    assert in_top2 == [0.0, 0.0, 1.0, 0.0]
    assert outside_top2 == [0.0, 1.0, 0.0, 0.0]


def test_load_hidden_subset_cache(tmp_path) -> None:
    npz_path = tmp_path / "hidden.npz"
    meta_path = tmp_path / "hidden.json"
    features = np.arange(3 * 4 * 1 * 5, dtype=np.float32).reshape(3, 4, 1, 5)
    np.savez_compressed(npz_path, features=features)
    meta_path.write_text(json.dumps({"row_ids": ["a", "b", "c"]}), encoding="utf-8")

    loaded = gate._load_hidden_subset_cache(npz_path=npz_path, meta_path=meta_path, rows=[_Row("c"), _Row("a")])

    assert loaded is not None
    subset, metadata = loaded
    assert subset.shape == (2, 4, 1, 5)
    assert metadata["subset_row_count"] == 2
    assert np.allclose(subset[0], features[2])


def _prediction_row(row_id: str, condition: str, prediction: int, answer: int, kl: float) -> dict:
    return {
        "row_id": row_id,
        "content_id": row_id,
        "condition": condition,
        "answer_index": answer,
        "answer_label": chr(ord("A") + answer),
        "prediction_index": prediction,
        "prediction_label": chr(ord("A") + prediction),
        "correct": prediction == answer,
        "full_prompt_prediction_index": prediction,
        "full_prompt_prediction_label": chr(ord("A") + prediction),
        "agrees_with_full_prompt": True,
        "margin": 0.2,
        "kl_to_full": kl,
        "scores": [float(prediction == index) for index in range(4)],
    }


def test_condition_metrics_cover_source_hidden_controls() -> None:
    rows = []
    full_predictions = [0, 1, 2, 3]
    answers = [0, 1, 0, 2]
    for index, (full_prediction, answer) in enumerate(zip(full_predictions, answers, strict=True)):
        row_id = f"r{index}"
        for condition in gate.CONDITIONS:
            prediction = full_prediction if condition in {"full_prompt", "source_hidden_residual_slots"} else 0
            rows.append(_prediction_row(row_id, condition, prediction, answer, 0.05))

    metrics = gate._condition_metrics(rows, seed=17, bootstrap_samples=25)

    assert metrics["source_hidden_residual_slots"]["agreement_with_full_prompt"] == 1.0
    assert metrics["wrong_source_hidden"]["paired_vs_full_prompt_accuracy"]["samples"] == 25
