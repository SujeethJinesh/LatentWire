from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_qwen_to_phi_harm_controlled_bucket_gate as gate


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _base_row(index: int) -> dict:
    answer = 1 if index % 2 == 0 else 0
    row = {
        "row_id": f"row-{index}",
        "answer_index": answer,
        "selected_prediction": 0,
        "hidden_mean_prediction": 0,
        "score_mean_prediction": 0,
        "vote_prediction": 0,
        "score_vote_prediction": 0,
        "selected_margin": 1.0 if answer == 1 else 0.1,
        "source_label_prediction": 1,
        "source_rank_only_bagged_prediction": 1,
        "score_only_bagged_prediction": 1,
        "trained_label_prediction": 1,
        "wrong_example_hidden_prediction": 2,
        "zero_hidden_prediction": 0,
        "candidate_roll_hidden_prediction": 1,
        "score_channel_roll_hidden_prediction": 2,
    }
    row["qwen_hybrid_prediction"] = 0
    return row


def _synthetic_calibration(row_count: int = 48) -> dict:
    answers = np.asarray([1 if index % 2 == 0 else 0 for index in range(row_count)], dtype=np.int64)
    hybrid = np.zeros(row_count, dtype=np.int64)
    mean = hybrid.copy()
    qwen_scores = np.full((row_count, 4), -1.0, dtype=np.float64)
    margin = np.full(row_count, 0.1, dtype=np.float64)
    for index, answer in enumerate(answers):
        if int(answer) == 1:
            qwen_scores[index] = [0.2, 1.8, -0.6, -0.7]
            margin[index] = 1.7
        else:
            qwen_scores[index] = [1.8, 0.2, -0.6, -0.7]
    return {
        "rows": [{"row_id": f"train-{index}", "answer_index": int(answers[index])} for index in range(row_count)],
        "answers": answers,
        "scores": qwen_scores,
        "hybrid": hybrid,
        "mean": mean,
        "margin": margin,
        "duplicate_row_count": 0,
        "oob_overlap_drop_count": 0,
        "sample_cache_rows": [],
        "component_rows": [],
    }


def _synthetic_phi_scores(answers: np.ndarray) -> np.ndarray:
    scores = np.full((len(answers), 4), -1.0, dtype=np.float64)
    for index, answer in enumerate(answers):
        scores[index, 0] = 0.6
        scores[index, int(answer)] = 1.9
    return scores


def test_bucket_model_accepts_only_low_harm_bucket():
    calibration = _synthetic_calibration()
    phi_scores = _synthetic_phi_scores(calibration["answers"])
    signals = gate._base_signals(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
    )
    fit_indices = np.arange(0, 32)
    dev_indices = np.arange(32, 48)
    bins = gate._fit_bins(signals, fit_indices)
    actions, fields = gate._action_fields(signals, bins)
    model, config_rows, bucket_rows = gate._select_bucket_model(
        actions=actions,
        fields=fields,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        bootstrap_samples=50,
    )
    predictions = gate._predict_bucket_model(
        fields={key: value[dev_indices] for key, value in fields.items()},
        actions=actions[dev_indices],
        hybrid=calibration["hybrid"][dev_indices],
        model=model,
    )
    assert config_rows
    assert bucket_rows
    assert float(np.mean(predictions == calibration["answers"][dev_indices])) > float(
        np.mean(calibration["hybrid"][dev_indices] == calibration["answers"][dev_indices])
    )


def test_action_fields_are_integer_quantized_packet_fields():
    calibration = _synthetic_calibration(12)
    phi_scores = _synthetic_phi_scores(calibration["answers"])
    signals = gate._base_signals(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
    )
    bins = gate._fit_bins(signals, np.arange(12))
    actions, fields = gate._action_fields(signals, bins)
    assert actions.shape == (12, 2)
    for key in ("q_margin_bin", "q_rival_adv_bin", "selected_margin_bin", "phi_margin_bin", "phi_action_adv_bin"):
        assert np.issubdtype(fields[key].dtype, np.integer)
        assert int(np.max(fields[key])) <= 4


def test_build_gate_writes_harm_controlled_artifacts(tmp_path, monkeypatch):
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
                "source_predictions": [row["answer_index"] for row in rows],
                "source_scores": [
                    [float(candidate == row["answer_index"]) for candidate in range(4)]
                    for row in rows
                ],
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
                "source_predictions": [1 if row["answer_index"] == 1 else 0 for row in rows],
                "source_scores": [
                    [0.2, 1.8, -0.6, -0.7] if row["answer_index"] == 1 else [1.8, 0.2, -0.6, -0.7]
                    for row in rows
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calibration = _synthetic_calibration()
    phi_scores = _synthetic_phi_scores(calibration["answers"])
    monkeypatch.setattr(gate.source_gate, "_build_qwen_oob_calibration", lambda **_: calibration)
    monkeypatch.setattr(
        gate.receiver_gate,
        "_arc_rows_for_calibration",
        lambda **_: ([type("Row", (), {"content_id": f"c{i}"})() for i in range(len(calibration["rows"]))], {}),
    )
    monkeypatch.setattr(
        gate.receiver_gate,
        "_load_or_build_phi_scores",
        lambda **_: (
            phi_scores,
            np.argmax(phi_scores, axis=1),
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
        bootstrap_samples=50,
        run_date="2026-05-04",
    )
    assert payload["headline"]["eval_rows"] == 12
    assert payload["packet_contract"]["raw_scores_or_logits_transmitted"] is False
    assert payload["packet_contract"]["raw_qwen_scores_used_only_for_source_side_quantized_packet"] is True
    method_names = {row["method"] for row in payload["method_rows"]}
    assert "harm_controlled_bucket_accept_defer_packet" in method_names
    assert "source_score_row_shuffle_before_encoding_bucket_control" in method_names
    assert (tmp_path / "out" / "hellaswag_qwen_to_phi_harm_controlled_bucket_gate.json").exists()
