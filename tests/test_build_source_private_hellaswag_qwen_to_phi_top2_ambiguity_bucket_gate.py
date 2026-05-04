from __future__ import annotations

import json

import numpy as np

from scripts import build_source_private_hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate as gate


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
        "selected_margin": 1.8 if answer == 1 else 0.1,
        "source_label_prediction": 1,
        "source_rank_only_bagged_prediction": 1,
        "score_only_bagged_prediction": 1,
        "trained_label_prediction": 1,
        "wrong_example_hidden_prediction": 2,
        "zero_hidden_prediction": 0,
        "candidate_roll_hidden_prediction": 1,
        "score_channel_roll_hidden_prediction": 2,
    }


def _synthetic_calibration(row_count: int = 64) -> dict:
    answers = np.asarray([1 if index % 2 == 0 else 0 for index in range(row_count)], dtype=np.int64)
    hybrid = np.zeros(row_count, dtype=np.int64)
    mean = hybrid.copy()
    qwen_scores = np.full((row_count, 4), -1.0, dtype=np.float64)
    margin = np.full(row_count, 0.1, dtype=np.float64)
    for index, answer in enumerate(answers):
        if int(answer) == 1:
            qwen_scores[index] = [0.2, 2.2, 0.3, -0.8]
            margin[index] = 1.8
        else:
            qwen_scores[index] = [2.1, 0.1, 0.0, -0.8]
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
        scores[index, 0] = 0.8
        scores[index, int(answer)] = 1.6
    return scores


def test_action_fields_expose_top2_packet_and_zeroable_syndrome_bins() -> None:
    calibration = _synthetic_calibration(16)
    phi_scores = _synthetic_phi_scores(calibration["answers"])
    bins = gate._fit_bins(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_margin=calibration["margin"],
        fit_indices=np.arange(16),
    )

    actions, fields, diagnostics = gate._action_fields(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
        bins=bins,
    )
    zero_actions, zero_fields, _ = gate._action_fields(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
        bins=bins,
        zero_source_bins=True,
    )

    assert actions.shape == (16, len(gate.ACTION_NAMES))
    assert np.array_equal(actions, zero_actions)
    assert set(np.unique(diagnostics["source_top1"])).issubset({0, 1})
    assert np.all(zero_fields["q_margin_bin"] == 0)
    assert np.all(zero_fields["q_entropy_bin"] == 0)
    assert np.any(fields["q_margin_bin"] != 0)


def test_bucket_selector_can_learn_synthetic_top2_override() -> None:
    calibration = _synthetic_calibration(64)
    phi_scores = _synthetic_phi_scores(calibration["answers"])
    fit_indices = np.arange(0, 48)
    dev_indices = np.arange(48, 64)
    bins = gate._fit_bins(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_margin=calibration["margin"],
        fit_indices=fit_indices,
    )
    actions, fields, _ = gate._action_fields(
        qwen_scores=calibration["scores"],
        phi_scores=phi_scores,
        hybrid=calibration["hybrid"],
        qwen_mean=calibration["mean"],
        qwen_margin=calibration["margin"],
        bins=bins,
    )

    model, config_rows, _ = gate._select_model(
        actions=actions,
        fields=fields,
        hybrid=calibration["hybrid"],
        answers=calibration["answers"],
        fit_indices=fit_indices,
        dev_indices=dev_indices,
        bootstrap_samples=50,
    )
    predictions = gate.bucket_gate._predict_bucket_model(
        fields={key: value[dev_indices] for key, value in fields.items()},
        actions=actions[dev_indices],
        hybrid=calibration["hybrid"][dev_indices],
        model=model,
    )

    assert config_rows
    assert float(np.mean(predictions == calibration["answers"][dev_indices])) > float(
        np.mean(calibration["hybrid"][dev_indices] == calibration["answers"][dev_indices])
    )


def test_build_gate_writes_top2_ambiguity_artifacts(tmp_path, monkeypatch):
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
                    [0.2, 2.2, 0.3, -0.8] if row["answer_index"] == 1 else [2.1, 0.1, 0.0, -0.8]
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
        lambda **_: ([type("Row", (), {"content_id": f"train-c{i}"})() for i in range(len(calibration["rows"]))], {}),
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
    assert payload["headline"]["source_top1_or_top2_oracle_accuracy"] >= payload["headline"]["fixed_hybrid_accuracy"]
    method_names = {row["method"] for row in payload["method_rows"]}
    assert "top2_ambiguity_bucket_packet" in method_names
    assert "source_pair_no_syndrome_bucket_control" in method_names
    assert "target_derived_source_packet_bucket_control" in method_names
    assert "raw_source_score_logit_fusion_control" in method_names
    assert (tmp_path / "out" / "hellaswag_qwen_to_phi_top2_ambiguity_bucket_gate.json").exists()
