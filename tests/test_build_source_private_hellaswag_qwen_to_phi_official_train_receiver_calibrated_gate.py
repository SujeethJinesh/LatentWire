from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate as gate


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _base_row(index: int) -> dict:
    answer = index % 4
    hybrid = answer if index % 3 else (answer + 1) % 4
    target = answer if index % 5 == 0 else (answer + 2) % 4
    row = {
        "row_id": f"row-{index}",
        "answer_index": answer,
        "selected_prediction": hybrid,
        "hidden_mean_prediction": hybrid,
        "score_mean_prediction": hybrid if index % 2 else (hybrid + 1) % 4,
        "vote_prediction": hybrid,
        "score_vote_prediction": hybrid,
        "selected_margin": float((index % 7) / 3.0),
        "source_label_prediction": (hybrid + 1) % 4,
        "source_rank_only_bagged_prediction": (hybrid + 2) % 4,
        "score_only_bagged_prediction": (hybrid + 3) % 4,
        "trained_label_prediction": (hybrid + 1) % 4,
        "wrong_example_hidden_prediction": (hybrid + 2) % 4,
        "zero_hidden_prediction": 0,
        "candidate_roll_hidden_prediction": (hybrid + 1) % 4,
        "score_channel_roll_hidden_prediction": (hybrid + 2) % 4,
        "_target_for_cache": target,
    }
    row["qwen_hybrid_prediction"] = hybrid
    return row


def _synthetic_calibration(row_count: int = 24) -> dict:
    answers = np.asarray([1 if index % 2 == 0 else 0 for index in range(row_count)], dtype=np.int64)
    hybrid = np.zeros(row_count, dtype=np.int64)
    mean = hybrid.copy()
    qwen_scores = np.full((row_count, 4), -1.0, dtype=np.float64)
    margin = np.ones(row_count, dtype=np.float64)
    for index in range(row_count):
        if answers[index] == 1:
            qwen_scores[index] = [0.2, 1.5, -0.5, -0.8]
        else:
            qwen_scores[index] = [1.4, 0.1, -0.4, -0.9]
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
        scores[index, int(answer)] = 1.7 if index % 3 == 0 else 0.3
        scores[index, 0] += 0.5
    return scores


def test_receiver_fit_uses_phi_side_information():
    calibration = _synthetic_calibration()
    phi_scores = _synthetic_phi_scores(calibration["answers"])
    action_features, action_candidates, _ = gate._stack_action_features(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
    )
    model, rows = gate._fit_receiver(
        action_features=action_features,
        action_candidates=action_candidates,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        fit_indices=np.arange(0, 18),
        dev_indices=np.arange(18, 24),
        ridges=(0.001, 1.0),
        bootstrap_samples=50,
    )
    predictions = gate._predict_receiver(action_features, action_candidates, calibration["hybrid"], model)
    assert predictions.shape == calibration["answers"].shape
    assert float(np.mean(predictions == calibration["answers"])) >= float(
        np.mean(calibration["hybrid"] == calibration["answers"])
    )
    assert rows[0]["official_dev_accuracy"] >= rows[-1]["official_dev_accuracy"]


def test_source_corruption_controls_preserve_shape():
    scores = np.arange(24, dtype=np.float64).reshape(6, 4)
    hybrid = np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int64)
    mean = hybrid.copy()
    margin = np.ones(6, dtype=np.float64)
    for condition in ("matched", "source_row_shuffle", "candidate_roll_source", "code_value_permutation"):
        c_scores, c_hybrid, c_mean, c_margin = gate._permuted_source_inputs(
            qwen_scores=scores,
            hybrid=hybrid,
            mean=mean,
            margin=margin,
            condition=condition,
            seed=7,
        )
        assert c_scores.shape == scores.shape
        assert c_hybrid.shape == hybrid.shape
        assert c_mean.shape == mean.shape
        assert c_margin.shape == margin.shape
        assert set(c_hybrid.tolist()).issubset({0, 1, 2, 3})


def test_build_gate_writes_receiver_calibrated_artifacts(tmp_path, monkeypatch):
    qwen_path = tmp_path / "qwen.jsonl"
    phi_path = tmp_path / "phi.json"
    score_path = tmp_path / "qwen_scores.json"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")
    rows = [_base_row(index) for index in range(16)]
    _write_jsonl(qwen_path, [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows])
    phi_path.write_text(
        json.dumps(
            {
                "row_count": len(rows),
                "row_ids": [row["row_id"] for row in rows],
                "source_predictions": [row["_target_for_cache"] for row in rows],
                "source_scores": [
                    [float(candidate == row["_target_for_cache"]) for candidate in range(4)]
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
                "source_predictions": [int(row["qwen_hybrid_prediction"]) for row in rows],
                "source_scores": [
                    [
                        float(candidate == row["qwen_hybrid_prediction"])
                        + 0.4 * float(candidate == ((row["qwen_hybrid_prediction"] + 1) % 4))
                        + 0.01 * index
                        for candidate in range(4)
                    ]
                    for index, row in enumerate(rows)
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
        gate,
        "_arc_rows_for_calibration",
        lambda **_: ([type("Row", (), {"content_id": f"c{i}"})() for i in range(len(calibration["rows"]))], {}),
    )
    monkeypatch.setattr(
        gate,
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
        fit_rows_per_slice=4,
        select_rows_per_slice=4,
        bootstrap_samples=50,
        run_date="2026-05-04",
    )
    assert payload["headline"]["eval_rows"] == 8
    assert payload["packet_contract"]["phi_official_train_scores_used_for_training"] is True
    method_names = {row["method"] for row in payload["method_rows"]}
    assert "official_train_receiver_calibrated_packet" in method_names
    assert "source_row_shuffle_receiver_control" in method_names
    assert (tmp_path / "out" / "hellaswag_qwen_to_phi_official_train_receiver_calibrated_gate.json").exists()
