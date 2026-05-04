from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from scripts import build_source_private_hellaswag_qwen_to_phi_quantized_score_packet_gate as gate


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _base_row(index: int) -> dict:
    answer = 1 if index % 2 == 0 else 0
    return {
        "row_id": f"row-{index}",
        "content_id": f"eval-{index}",
        "answer_index": answer,
        "selected_prediction": 0,
        "hidden_mean_prediction": 0,
        "score_mean_prediction": 0,
        "vote_prediction": 0,
        "score_vote_prediction": 0,
        "selected_margin": 0.2,
        "source_rank_only_bagged_prediction": 0,
        "score_only_bagged_prediction": 0,
        "trained_label_prediction": 0,
        "wrong_example_hidden_prediction": 2,
        "zero_hidden_prediction": 0,
        "candidate_roll_hidden_prediction": 1,
        "score_channel_roll_hidden_prediction": 2,
    }


def _qwen_scores_for_answer(answer: int) -> list[float]:
    if int(answer) == 1:
        return [0.2, 2.4, 0.0, -0.8]
    return [2.4, 0.2, 0.0, -0.8]


def _synthetic_calibration(row_count: int = 64) -> dict:
    answers = np.asarray([1 if index % 2 == 0 else 0 for index in range(row_count)], dtype=np.int64)
    hybrid = np.zeros(row_count, dtype=np.int64)
    scores = np.asarray([_qwen_scores_for_answer(int(answer)) for answer in answers], dtype=np.float64)
    return {
        "rows": [{"row_id": f"train-{index}", "answer_index": int(answers[index])} for index in range(row_count)],
        "answers": answers,
        "scores": scores,
        "hybrid": hybrid,
        "mean": hybrid.copy(),
        "margin": np.full(row_count, 1.8, dtype=np.float64),
        "duplicate_row_count": 0,
        "oob_overlap_drop_count": 0,
        "sample_cache_rows": [],
        "component_rows": [],
    }


def _synthetic_phi_scores(row_count: int) -> np.ndarray:
    scores = np.full((row_count, 4), -1.0, dtype=np.float64)
    scores[:, 0] = 1.4
    scores[:, 1] = 0.6
    return scores


def test_quantized_score_reconstruction_preserves_shape_and_rotation_is_finite() -> None:
    scores = np.asarray(
        [
            [0.2, 2.4, 0.0, -0.8],
            [2.4, 0.2, 0.0, -0.8],
            [0.0, -0.2, 1.7, 0.1],
        ],
        dtype=np.float64,
    )
    rotation = gate._orthogonal_matrix(4, seed=17)

    uniform = gate._reconstruct_scores(scores, codec="uniform_zscore", raw_payload_bytes=1, clip=2.5)
    rotated = gate._reconstruct_scores(
        scores,
        codec="rotated_uniform_zscore",
        raw_payload_bytes=2,
        clip=2.5,
        rotation=rotation,
    )

    assert uniform.shape == scores.shape
    assert rotated.shape == scores.shape
    assert np.isfinite(uniform).all()
    assert np.isfinite(rotated).all()
    assert gate._codec_bits_per_coord(1) == 2
    assert gate._codec_bits_per_coord(8) == 16
    assert gate._framed_bytes(4) == 7


def test_fit_model_can_learn_synthetic_quantized_score_override() -> None:
    calibration = _synthetic_calibration()
    phi_scores = _synthetic_phi_scores(len(calibration["answers"]))
    reconstructed = gate._reconstruct_scores(
        calibration["scores"],
        codec="uniform_zscore",
        raw_payload_bytes=2,
        clip=2.5,
    )
    dev_indices = np.arange(16, 64)

    model, config_rows = gate._fit_model(
        reconstructed_qwen=reconstructed,
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        dev_indices=dev_indices,
        bootstrap_samples=50,
    )
    predictions = gate._predict_blend(
        reconstructed_qwen=reconstructed[dev_indices],
        phi_scores=phi_scores[dev_indices],
        hybrid=calibration["hybrid"][dev_indices],
        model=model,
    )

    assert config_rows
    assert float(np.mean(predictions == calibration["answers"][dev_indices])) > float(
        np.mean(calibration["hybrid"][dev_indices] == calibration["answers"][dev_indices])
    )


def test_build_gate_writes_quantized_score_packet_artifacts(tmp_path, monkeypatch):
    qwen_path = tmp_path / "qwen.jsonl"
    phi_path = tmp_path / "phi.json"
    score_path = tmp_path / "qwen_scores.json"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")
    rows = [_base_row(index) for index in range(24)]
    _write_jsonl(qwen_path, rows)
    phi_path.write_text(
        json.dumps(
            {
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_predictions": [0 for _ in rows],
                "source_scores": _synthetic_phi_scores(len(rows)).tolist(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    score_path.write_text(
        json.dumps(
            {
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_model": {"name": "synthetic"},
                "source_predictions": [row["answer_index"] for row in rows],
                "source_scores": [_qwen_scores_for_answer(row["answer_index"]) for row in rows],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calibration = _synthetic_calibration()
    monkeypatch.setattr(gate.source_gate, "_build_qwen_oob_calibration", lambda **_: calibration)
    monkeypatch.setattr(
        gate.receiver_gate,
        "_arc_rows_for_calibration",
        lambda **_: ([SimpleNamespace(content_id=f"train-c{i}") for i in range(len(calibration["rows"]))], {}),
    )
    monkeypatch.setattr(
        gate.receiver_gate,
        "_load_or_build_phi_scores",
        lambda **_: (
            _synthetic_phi_scores(len(calibration["rows"])),
            np.zeros(len(calibration["rows"]), dtype=np.int64),
            {"cache_hit": True, "kind": "synthetic"},
            "synthetic-sha",
        ),
    )

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        train_path=train_path,
        qwen_train_cache_dir=tmp_path / "cache",
        slices=(
            {
                "slice_start": 0,
                "slice_end_exclusive": len(rows),
                "qwen_predictions": qwen_path,
                "phi_target_score_cache": phi_path,
            },
        ),
        source_score_cache=score_path,
        fit_rows_per_slice=6,
        select_rows_per_slice=6,
        budget_bytes=(1, 2),
        bootstrap_samples=50,
        run_date="2026-05-04",
    )

    assert payload["headline"]["eval_rows"] == 12
    assert payload["packet_contract"]["raw_scores_or_logits_transmitted"] is False
    assert payload["packet_contract"]["quantized_source_score_vector_transmitted"] is True
    assert payload["budget_rows"]
    assert payload["headline"]["best_quantized_accuracy"] >= payload["headline"]["fixed_hybrid_accuracy"]
    method_names = {row["method"] for row in payload["method_rows"]}
    assert "quantized_score_packet_uniform_zscore_1B" in method_names
    assert "source_row_shuffle_uniform_zscore_1B_control" in method_names
    assert "target_derived_source_packet_uniform_zscore_1B_control" in method_names
    assert (tmp_path / "out" / "hellaswag_qwen_to_phi_quantized_score_packet_gate.json").exists()
    assert (tmp_path / "out" / "budget_rows.csv").exists()
