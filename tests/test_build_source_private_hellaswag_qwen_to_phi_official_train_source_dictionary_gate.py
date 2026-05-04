from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_qwen_to_phi_official_train_source_dictionary_gate as gate


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
    scores = np.full((row_count, 4), -1.0, dtype=np.float64)
    margin = np.zeros(row_count, dtype=np.float64)
    for index in range(row_count):
        if answers[index] == 1:
            scores[index] = [0.2, 1.5, -0.5, -0.8]
            margin[index] = 1.3
        else:
            scores[index] = [1.4, 0.1, -0.4, -0.9]
            margin[index] = 1.3
    return {
        "rows": [{"row_id": f"train-{index}", "answer_index": int(answers[index])} for index in range(row_count)],
        "answers": answers,
        "scores": scores,
        "hybrid": hybrid,
        "mean": mean,
        "margin": margin,
        "duplicate_row_count": 0,
        "oob_overlap_drop_count": 0,
        "sample_cache_rows": [],
        "component_rows": [],
    }


def test_pair_features_and_fit_dictionary_learns_useful_rival():
    calibration = _synthetic_calibration()
    pair = gate._pair_from_scores(calibration["scores"], calibration["hybrid"])
    assert pair.shape == (24, 4)
    assert np.all(pair[:, 0] == 0)
    features = gate._feature_matrix(
        scores=calibration["scores"],
        hybrid=calibration["hybrid"],
        mean_prediction=calibration["mean"],
        margin=calibration["margin"],
    )
    assert features.shape[0] == 24
    model, rows = gate._fit_dictionary(
        features=features,
        pair=pair,
        answers=calibration["answers"],
        fit_indices=np.arange(0, 18),
        dev_indices=np.arange(18, 24),
        ridges=(0.001, 1.0),
        bootstrap_samples=50,
    )
    predictions = gate._predict_dictionary(features, pair, model)
    fixed_accuracy = float(np.mean(pair[:, 0] == calibration["answers"]))
    selected_accuracy = float(np.mean(predictions == calibration["answers"]))
    assert selected_accuracy >= fixed_accuracy
    assert rows[0]["threshold_is_noop"] in {True, False}


def test_source_dictionary_controls_are_same_length_and_candidate_valid():
    predictions = np.asarray([0, 1, 1, 2, 3, 0], dtype=np.int64)
    for condition in ("matched", "source_row_shuffle", "random_same_byte", "code_value_permutation", "candidate_roll_code"):
        corrupted = gate._codes_for_condition(predictions, condition=condition, seed=17)
        assert corrupted.shape == predictions.shape
        assert set(corrupted.tolist()).issubset({0, 1, 2, 3})
    assert np.all(gate._codes_for_condition(predictions, condition="candidate_roll_code", seed=17) == (predictions + 1) % 4)


def test_build_gate_writes_official_train_source_dictionary_artifacts(tmp_path, monkeypatch):
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
    monkeypatch.setattr(gate, "_build_qwen_oob_calibration", lambda **_: _synthetic_calibration())
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
    assert payload["packet_contract"]["source_text_exposed"] is False
    assert payload["source_score_metadata"]["source_score_cache_rows"] == len(rows)
    method_names = {row["method"] for row in payload["method_rows"]}
    assert "official_train_source_dictionary_packet" in method_names
    assert "source_row_shuffle_source_dictionary_control" in method_names
    assert "official_train_label_permutation_dictionary_control" in method_names
    assert payload["packet_accounting"]["raw_payload_bits"] == 2
    assert (tmp_path / "out" / "hellaswag_qwen_to_phi_official_train_source_dictionary_gate.json").exists()
