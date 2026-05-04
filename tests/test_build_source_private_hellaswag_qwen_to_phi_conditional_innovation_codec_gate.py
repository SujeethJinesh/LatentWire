from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_qwen_to_phi_conditional_innovation_codec_gate as gate


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
        scores[index, int(answer)] = 1.6 if index % 3 == 0 else 0.3
        scores[index, 0] += 0.4
    return scores


def test_innovation_code_controls_preserve_shape() -> None:
    rows = []
    for index in range(8):
        answer = index % 4
        row = _base_row(index)
        row["phi_target_prediction"] = (answer + 1) % 4
        row["phi_target_scores"] = [float(candidate == row["phi_target_prediction"]) for candidate in range(4)]
        row["qwen_source_scores"] = [float(candidate == answer) + 0.1 * candidate for candidate in range(4)]
        rows.append(row)
    qwen_scores = gate._source_scores(rows)
    phi_scores = gate._phi_scores(rows)
    codec = gate._fit_codec_state(qwen_scores=qwen_scores, phi_scores=phi_scores, fit_indices=np.arange(4))

    matched = gate._encode_innovation_codes(rows=rows, codec=codec, condition="matched", seed=7)
    for condition in (
        "source_row_shuffle",
        "random_same_byte",
        "code_value_permutation",
        "candidate_roll_code",
        "target_derived",
    ):
        corrupted = gate._encode_innovation_codes(rows=rows, codec=codec, condition=condition, seed=7)
        assert set(corrupted) == set(matched)
        for rate in matched:
            assert corrupted[rate].shape == matched[rate].shape


def test_build_gate_writes_conditional_innovation_artifacts(tmp_path, monkeypatch) -> None:
    qwen_path = tmp_path / "qwen.jsonl"
    phi_path = tmp_path / "phi.json"
    score_path = tmp_path / "qwen_scores.json"
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("{}\n", encoding="utf-8")
    rows = [_base_row(index) for index in range(16)]
    _write_jsonl(qwen_path, [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows])
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
        phi_train_score_cache=tmp_path / "phi_train_scores.json",
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
        rate_bytes=(1, 2),
        bootstrap_samples=50,
        run_date="2026-05-04",
    )

    assert payload["headline"]["eval_rows"] == 8
    assert payload["packet_contract"]["qwen_scores_used_only_for_source_side_residual_quantization"] is True
    assert payload["packet_contract"]["phi_scores_used_as_receiver_side_information"] is True
    method_names = {row["method"] for row in payload["method_rows"]}
    assert "conditional_innovation_packet" in method_names
    assert "ghost_only_receiver_control" in method_names
    assert "target_derived_innovation_control" in method_names
    assert (tmp_path / "out" / "hellaswag_qwen_to_phi_conditional_innovation_codec_gate.json").exists()
