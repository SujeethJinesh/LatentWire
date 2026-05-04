from __future__ import annotations

import numpy as np

from scripts import build_source_private_hellaswag_qwen_to_phi_top2_rival_codebook_gate as gate


def _row(answer: int, hybrid: int, phi_pred: int, q_scores: list[float], phi_scores: list[float]) -> dict:
    return {
        "row_id": str(len(q_scores)) + str(answer),
        "answer_index": answer,
        "qwen_hybrid_prediction": hybrid,
        "selected_prediction": hybrid,
        "phi_target_prediction": phi_pred,
        "qwen_source_scores": q_scores,
        "phi_target_scores": phi_scores,
    }


def test_action_table_uses_top2_and_phi_actions() -> None:
    rows = [_row(1, 0, 3, [0.2, 1.5, 0.9, -0.1], [0.1, 0.3, 0.0, 1.1])]
    bins = gate._fit_bins(rows)

    actions, fields = gate._action_table(rows, bins=bins)

    assert actions.shape == (1, 4)
    assert list(actions[0]) == [0, 1, 2, 3]
    assert len(fields[0]) == 4


def test_bucket_model_can_override_hybrid() -> None:
    fit_rows = [
        _row(1, 0, 3, [0.1, 2.0, 1.0, -1.0], [1.0, 0.2, 0.0, 0.1]),
        _row(1, 0, 3, [0.2, 2.1, 1.1, -1.0], [1.0, 0.2, 0.0, 0.1]),
        _row(2, 0, 3, [0.1, 1.0, 2.0, -1.0], [1.0, 0.0, 0.2, 0.1]),
        _row(2, 0, 3, [0.2, 1.1, 2.1, -1.0], [1.0, 0.0, 0.2, 0.1]),
    ]
    bins = gate._fit_bins(fit_rows)
    buckets = gate._fit_bucket_stats(
        rows=fit_rows,
        bins=bins,
        min_support=2,
        min_mean_delta=0.0,
        max_harm_rate=0.0,
    )

    predictions = gate._predict_with_buckets(fit_rows, bins=bins, buckets=buckets)

    assert buckets
    assert np.mean(predictions == gate._answers(fit_rows)) > np.mean(
        gate._field_array(fit_rows, "qwen_hybrid_prediction") == gate._answers(fit_rows)
    )


def test_paired_ci_reports_samples() -> None:
    selected = np.asarray([1, 1, 0])
    baseline = np.asarray([0, 1, 0])
    answers = np.asarray([1, 0, 0])

    ci = gate._paired_ci(selected=selected, baseline=baseline, answers=answers, seed=5, samples=25)

    assert ci["helps"] >= 0
    assert "ci95_low" in ci
