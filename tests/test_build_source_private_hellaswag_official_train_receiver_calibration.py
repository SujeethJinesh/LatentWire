from __future__ import annotations

import numpy as np
import pytest

from scripts import build_source_private_hellaswag_official_train_receiver_calibration as gate


def _sample(row_ids: list[str], answers: list[int], selected: list[int]) -> dict[str, object]:
    scores = np.zeros((len(row_ids), 4), dtype=np.float64)
    hidden = np.zeros((len(row_ids), 4, 3), dtype=np.float32)
    for row_index, prediction in enumerate(selected):
        scores[row_index, prediction] = 1.0
        hidden[row_index, prediction, 0] = 1.0
    return {
        "row_ids": row_ids,
        "content_digest": "shared",
        "answers": np.asarray(answers, dtype=np.int64),
        "scores": scores,
        "hidden": hidden,
        "receiver_hidden": hidden,
        "mock_selected": np.asarray(selected, dtype=np.int64),
    }


def _fake_packet_predictions_for_sample(*, sample, model_bank, included_seeds, aggregation_policy):
    del model_bank, included_seeds, aggregation_policy
    selected = np.asarray(sample["mock_selected"], dtype=np.int64)
    return {
        "selected_prediction": selected,
        "selected_margin": np.linspace(0.1, 0.3, num=len(selected), dtype=np.float64),
        "mean_zscore_prediction": selected,
        "hybrid_vote_on_score_agreement_prediction": ((selected + 1) % 4).astype(np.int64),
        "score_mean_prediction": selected,
        "score_vote_prediction": selected,
        "vote_prediction": selected,
    }


def test_oob_calibration_rows_drop_rows_seen_by_included_models(monkeypatch) -> None:
    monkeypatch.setattr(gate, "_packet_predictions_for_sample", _fake_packet_predictions_for_sample)
    tiny_samples = {
        1: _sample(["a", "b"], [0, 1], [0, 1]),
        2: _sample(["b", "c"], [1, 2], [1, 2]),
    }
    qwen_samples = {
        1: _sample(["a", "b"], [0, 1], [3, 0]),
        2: _sample(["b", "c"], [1, 2], [1, 2]),
    }

    payload = gate._build_oob_calibration_rows(
        tiny_samples=tiny_samples,
        qwen_samples=qwen_samples,
        tiny_bank={},
        qwen_bank={},
        sample_seeds=(1, 2),
        tiny_aggregation_policy="mean_zscore",
    )

    assert [row["row_id"] for row in payload["rows"]] == ["a", "c"]
    assert payload["answers"].tolist() == [0, 2]
    assert payload["tiny_packet"].tolist() == [0, 2]
    assert payload["qwen_target"].tolist() == [3, 2]
    assert payload["qwen_hybrid"].tolist() == [0, 3]
    assert payload["oob_overlap_drop_count"] == 2
    assert payload["duplicate_row_count"] == 0


def test_oob_calibration_rows_reject_misaligned_family_caches(monkeypatch) -> None:
    monkeypatch.setattr(gate, "_packet_predictions_for_sample", _fake_packet_predictions_for_sample)
    tiny_samples = {1: _sample(["a", "b"], [0, 1], [0, 1])}
    qwen_samples = {1: _sample(["a", "x"], [0, 1], [0, 1])}

    with pytest.raises(ValueError, match="row IDs are not aligned"):
        gate._build_oob_calibration_rows(
            tiny_samples=tiny_samples,
            qwen_samples=qwen_samples,
            tiny_bank={},
            qwen_bank={},
            sample_seeds=(1,),
            tiny_aggregation_policy="mean_zscore",
        )
