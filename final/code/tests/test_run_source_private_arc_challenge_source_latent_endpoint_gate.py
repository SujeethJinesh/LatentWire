from __future__ import annotations

import json

import numpy as np

from scripts import run_source_private_arc_challenge_source_latent_endpoint_gate as endpoint


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_ridge_map_recovers_linear_source_to_target() -> None:
    source = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix = np.asarray([[2.0, -1.0], [0.5, 1.5]], dtype=np.float64)
    target = source @ matrix

    mapper = endpoint._fit_ridge_map(source, target, ridge=1e-6)
    pred = endpoint._apply_ridge_map(source, mapper)

    assert np.max(np.abs(pred - target)) < 1e-4
    assert mapper["source_dim"] == 2
    assert mapper["target_dim"] == 2


def test_source_latent_endpoint_gate_writes_controls(tmp_path) -> None:
    train_rows = [
        {
            "id": "train_hot",
            "question": "Which choice is hottest?",
            "choices": {"text": ["ice cube", "bright fire", "wet rain"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
        {
            "id": "train_cold",
            "question": "Which choice is coldest?",
            "choices": {"text": ["warm soup", "frozen ice", "open flame"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
        {
            "id": "train_water",
            "question": "Which choice is wet?",
            "choices": {"text": ["dry sand", "falling rain", "hot coal"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
    ]
    eval_rows = [
        {
            "id": "eval_hot",
            "question": "Which answer is hottest?",
            "choices": {"text": ["cold snow", "small fire", "blue water"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
        {
            "id": "eval_wet",
            "question": "Which answer is wet?",
            "choices": {"text": ["falling rain", "dry rock", "burning fire"], "label": ["A", "B", "C"]},
            "answerKey": "A",
        },
    ]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    payload = endpoint.run_gate(
        output_dir=tmp_path / "out",
        train_path=train_path,
        eval_path=eval_path,
        train_limit=None,
        eval_limit=None,
        budget_bytes=12,
        target_feature_dim=64,
        code_dim=32,
        target_feature_mode="hashed",
        target_feature_model="BAAI/bge-small-en",
        target_feature_device="cpu",
        target_feature_dtype="float32",
        target_feature_max_length=64,
        alignment_target_mode="target_residual",
        decoder_target_mode="target_residual",
        source_feature_mode="hashed_pair",
        source_feature_dim=64,
        source_lm_model="unused",
        source_lm_device="cpu",
        source_lm_dtype="float32",
        source_lm_max_length=64,
        source_lm_normalization="mean",
        source_hidden_layer=-1,
        source_score_mode="source_feature_ridge",
        source_score_ridge=0.1,
        alignment_ridge=0.1,
        local_files_only=True,
        seed=7,
        bootstrap_samples=25,
        min_lift_over_target=0.0,
        min_gap_over_control=0.0,
        min_gap_over_text=0.0,
    )

    assert payload["gate"] == "source_private_arc_challenge_source_latent_endpoint_gate"
    assert payload["train_eval_content_overlap_count"] == 0
    assert endpoint.MATCHED_CONDITION in payload["condition_metrics"]
    assert "source_feature_permutation_packet" in payload["condition_metrics"]
    assert payload["alignment"]["kind"] == "train_only_ridge_source_features_to_target_candidate_space"
    assert (tmp_path / "out" / "source_latent_endpoint_gate.json").exists()
    assert (tmp_path / "out" / "predictions.jsonl").exists()
