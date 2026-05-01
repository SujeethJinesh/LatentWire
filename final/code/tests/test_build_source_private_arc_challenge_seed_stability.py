from __future__ import annotations

import json

from scripts import build_source_private_arc_challenge_seed_stability as seed_stability
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _toy_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
            "id": "train_wet",
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
    return train_rows, eval_rows


def test_source_prediction_cache_rejects_missing_rows(tmp_path) -> None:
    row = arc_gate.ArcRow(
        row_id="r0",
        content_id="c0",
        question="Question?",
        choices=("alpha", "beta"),
        choice_labels=("A", "B"),
        answer_index=0,
        answer_label="A",
    )

    try:
        seed_stability._source_predictions_from_anchor([row], {})
    except ValueError as exc:
        assert "missing=1" in str(exc)
    else:
        raise AssertionError("missing anchor rows should fail")


def test_build_seed_stability_writes_answer_free_cache_and_summary(tmp_path) -> None:
    train_rows, eval_rows = _toy_rows()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    anchor = arc_gate.run_gate(
        output_dir=tmp_path / "anchor",
        train_path=train_path,
        eval_path=eval_path,
        train_limit=None,
        eval_limit=None,
        budget_bytes=12,
        feature_dim=64,
        code_dim=32,
        feature_mode="hashed",
        feature_model="BAAI/bge-small-en",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=64,
        local_files_only=True,
        source_score_mode="pair_ridge",
        source_lm_model="unused",
        source_lm_device="cpu",
        source_lm_dtype="float32",
        source_lm_max_length=64,
        source_lm_normalization="mean",
        ridge=0.1,
        seed=7,
        bootstrap_samples=25,
        min_lift_over_target=0.0,
        min_gap_over_control=0.0,
        min_gap_over_text=0.0,
    )
    payload = seed_stability.build_seed_stability(
        output_dir=tmp_path / "stable",
        train_path=train_path,
        eval_path=eval_path,
        anchor_predictions=tmp_path / "anchor" / "predictions.jsonl",
        split_name="toy",
        seeds=[7, 11],
        budget_bytes=12,
        feature_dim=64,
        code_dim=32,
        feature_mode="hashed",
        feature_model="BAAI/bge-small-en",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=64,
        local_files_only=True,
        bootstrap_samples=25,
        min_lift_over_target=0.0,
        min_gap_over_control=0.0,
        min_gap_over_text=0.0,
    )

    assert anchor["gate"] == "source_private_arc_challenge_fixed_packet_gate"
    assert payload["gate"] == "source_private_arc_challenge_seed_stability"
    assert payload["aggregate"]["seed_count"] == 2
    assert len(payload["per_seed"]) == 2
    assert (tmp_path / "stable" / "arc_challenge_seed_stability.json").exists()
    assert (tmp_path / "stable" / "per_seed_metrics.csv").exists()

    cache_rows = [
        json.loads(line)
        for line in (tmp_path / "stable" / "source_prediction_cache.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert cache_rows
    assert "answer_index" not in cache_rows[0]
    assert "answer_label" not in cache_rows[0]
    assert cache_rows[0]["source_visible_fields"] == ["question", "choices"]


def test_build_seed_stability_records_anchor_relative_basis(tmp_path) -> None:
    train_rows, eval_rows = _toy_rows()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    anchor = arc_gate.run_gate(
        output_dir=tmp_path / "anchor",
        train_path=train_path,
        eval_path=eval_path,
        train_limit=None,
        eval_limit=None,
        budget_bytes=12,
        feature_dim=32,
        code_dim=16,
        feature_mode="hashed",
        feature_model="BAAI/bge-small-en",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=64,
        local_files_only=True,
        source_score_mode="pair_ridge",
        source_lm_model="unused",
        source_lm_device="cpu",
        source_lm_dtype="float32",
        source_lm_max_length=64,
        source_lm_normalization="mean",
        ridge=0.1,
        seed=7,
        bootstrap_samples=25,
        min_lift_over_target=0.0,
        min_gap_over_control=0.0,
        min_gap_over_text=0.0,
    )
    payload = seed_stability.build_seed_stability(
        output_dir=tmp_path / "stable_anchor_relative",
        train_path=train_path,
        eval_path=eval_path,
        anchor_predictions=tmp_path / "anchor" / "predictions.jsonl",
        split_name="toy_anchor_relative",
        seeds=[7],
        budget_bytes=12,
        feature_dim=32,
        code_dim=16,
        feature_mode="anchor_relative_hashed",
        feature_model="BAAI/bge-small-en",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=64,
        local_files_only=True,
        bootstrap_samples=25,
        min_lift_over_target=0.0,
        min_gap_over_control=0.0,
        min_gap_over_text=0.0,
    )

    assert anchor["gate"] == "source_private_arc_challenge_fixed_packet_gate"
    assert payload["feature_mode"] == "anchor_relative_hashed"
    assert payload["anchor_relative_basis"]["anchor_source"] == "train split question/candidate texts"
    assert payload["anchor_relative_basis"]["base_feature_mode"] == "hashed"
    assert payload["anchor_control"] == "none"


def test_build_seed_stability_records_anchor_id_shuffle_control(tmp_path) -> None:
    train_rows, eval_rows = _toy_rows()
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    arc_gate.run_gate(
        output_dir=tmp_path / "anchor",
        train_path=train_path,
        eval_path=eval_path,
        train_limit=None,
        eval_limit=None,
        budget_bytes=12,
        feature_dim=32,
        code_dim=16,
        feature_mode="hashed",
        feature_model="BAAI/bge-small-en",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=64,
        local_files_only=True,
        source_score_mode="pair_ridge",
        source_lm_model="unused",
        source_lm_device="cpu",
        source_lm_dtype="float32",
        source_lm_max_length=64,
        source_lm_normalization="mean",
        ridge=0.1,
        seed=7,
        bootstrap_samples=25,
        min_lift_over_target=0.0,
        min_gap_over_control=0.0,
        min_gap_over_text=0.0,
    )
    payload = seed_stability.build_seed_stability(
        output_dir=tmp_path / "stable_anchor_id_shuffle",
        train_path=train_path,
        eval_path=eval_path,
        anchor_predictions=tmp_path / "anchor" / "predictions.jsonl",
        split_name="toy_anchor_id_shuffle",
        seeds=[7],
        budget_bytes=12,
        feature_dim=32,
        code_dim=16,
        feature_mode="anchor_relative_hashed",
        feature_model="BAAI/bge-small-en",
        feature_device="cpu",
        feature_dtype="float32",
        feature_max_length=64,
        local_files_only=True,
        bootstrap_samples=25,
        min_lift_over_target=0.0,
        min_gap_over_control=0.0,
        min_gap_over_text=0.0,
        anchor_control="anchor_id_shuffle",
    )

    assert payload["anchor_control"] == "anchor_id_shuffle"
    assert payload["anchor_control_metadata"]["anchor_control"] == "anchor_id_shuffle"
    assert payload["anchor_control_metadata"]["source_anchor_sha256"]
    assert payload["anchor_control_metadata"]["receiver_column_permutation_sha256"]
