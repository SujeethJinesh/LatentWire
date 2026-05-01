from __future__ import annotations

import json
from types import SimpleNamespace

from scripts import run_source_private_candidate_conditioned_residual_code_smoke as smoke


def _example(answer_index: int = 0):
    candidates = tuple(SimpleNamespace(label=f"candidate_{idx}") for idx in range(4))
    return SimpleNamespace(answer_label=f"candidate_{answer_index}", candidates=candidates)


def _row(
    *,
    condition: str,
    example_id: str,
    scores: list[float] | None,
    answer_index: int = 0,
    prior_index: int = 1,
    base_correct: bool = False,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "family_name": "fam",
        "budget_bytes": 8,
        "condition": condition,
        "answer": f"candidate_{answer_index}",
        "prediction": f"candidate_{answer_index if base_correct else prior_index}",
        "correct": base_correct,
        "strict_correct": base_correct,
        "answer_index": answer_index,
        "prior_index": prior_index,
        "payload_bytes": 8 if scores is not None else 0,
        "payload_tokens": 1 if scores is not None else 0,
        "payload_hex": "00" if scores is not None else "",
        "latency_ms": 0.0,
        "metadata": (
            {
                "scores": scores,
                "candidate_local_row_norms": [1.0, 1.0, 1.0, 1.0],
                "candidate_local_payload_l2": 1.0,
            }
            if scores is not None
            else {"decoder": "prior"}
        ),
    }


def test_fit_receiver_promotes_matched_and_suppresses_controls() -> None:
    train_rows = []
    for idx in range(16):
        train_rows.append(
            _row(
                condition=smoke.BASE_MATCHED_CONDITION,
                example_id=f"train_{idx}",
                scores=[0.9, 0.2, 0.1, 0.0],
                base_correct=False,
            )
        )
        train_rows.append(
            _row(
                condition="shuffled_source",
                example_id=f"train_{idx}",
                scores=[0.2, 0.8, 0.1, 0.0],
                base_correct=False,
            )
        )

    state = smoke._fit_receiver(train_rows, ridge=0.1, matched_weight=1.0, control_weight=1.0)
    example = _example()
    matched = smoke._convert_base_row(
        base_row=_row(
            condition=smoke.BASE_MATCHED_CONDITION,
            example_id="eval_0",
            scores=[0.85, 0.1, 0.0, -0.1],
            base_correct=False,
        ),
        display_condition=smoke.MATCHED_CONDITION,
        example=example,
        state=state,
        latency_ms=0.0,
    )
    control = smoke._convert_base_row(
        base_row=_row(
            condition="shuffled_source",
            example_id="eval_0",
            scores=[0.1, 0.85, 0.0, -0.1],
            base_correct=False,
        ),
        display_condition="shuffled_source",
        example=example,
        state=state,
        latency_ms=0.0,
    )

    assert matched["prediction_index"] == 0
    assert matched["correct"] is True
    assert control["prediction_index"] == 1
    assert control["correct"] is False
    assert matched["metadata"]["scores"]
    assert matched["metadata"]["raw_candidate_local_scores"] == [0.85, 0.1, 0.0, -0.1]


def test_prior_fallback_and_tie_prefers_prior() -> None:
    row = _row(condition=smoke.BASE_MATCHED_CONDITION, example_id="x", scores=[0.5, 0.5, 0.1, 0.0])
    prediction, delta = smoke._predict_index(row, [1.0, 1.0, 0.0, 0.0], threshold=0.0)
    assert prediction == 1
    assert delta == 0.0

    prediction, _ = smoke._predict_index(
        _row(condition="target_only", example_id="x", scores=None),
        None,
        threshold=0.0,
    )
    assert prediction == 1


def test_summary_requires_base_improvement_and_control_cleanliness() -> None:
    rows = []
    example = _example()
    state = smoke._fit_receiver(
        [
            _row(condition=smoke.BASE_MATCHED_CONDITION, example_id="train", scores=[0.9, 0.1, 0.0, 0.0]),
            _row(condition="shuffled_source", example_id="train", scores=[0.1, 0.9, 0.0, 0.0]),
        ],
        ridge=0.1,
        matched_weight=1.0,
        control_weight=1.0,
    )
    for idx in range(8):
        rows.append(
            smoke._convert_base_row(
                base_row=_row(condition="target_only", example_id=f"eval_{idx}", scores=None),
                display_condition="target_only",
                example=example,
                state=state,
                latency_ms=0.0,
            )
        )
        rows.append(
            smoke._convert_base_row(
                base_row=_row(
                    condition=smoke.BASE_MATCHED_CONDITION,
                    example_id=f"eval_{idx}",
                    scores=[0.9, 0.1, 0.0, 0.0],
                    base_correct=False,
                ),
                display_condition=smoke.MATCHED_CONDITION,
                example=example,
                state=state,
                latency_ms=0.0,
            )
        )
        for condition in smoke.STRICT_CONTROLS:
            rows.append(
                smoke._convert_base_row(
                    base_row=_row(condition=condition, example_id=f"eval_{idx}", scores=[0.1, 0.9, 0.0, 0.0]),
                    display_condition=condition,
                    example=example,
                    state=state,
                    latency_ms=0.0,
                )
            )
        rows.append(
            smoke._convert_base_row(
                base_row=_row(
                    condition=smoke.BASE_ORACLE_CONDITION,
                    example_id=f"eval_{idx}",
                    scores=[0.9, 0.1, 0.0, 0.0],
                    base_correct=True,
                ),
                display_condition=smoke.ORACLE_CONDITION,
                example=example,
                state=state,
                latency_ms=0.0,
            )
        )

    summary = smoke._summarize_rows(
        rows,
        direction="core_to_holdout",
        budget_bytes=8,
        seed=1,
        min_improvement_over_base=0.03,
        bootstrap_samples=20,
    )
    assert summary["exact_id_parity"] is True
    assert summary["matched_accuracy"] == 1.0
    assert summary["best_control_accuracy"] == 0.0
    assert summary["beats_base"] is True
    assert summary["pass_gate"] is True


def test_write_jsonl_preserves_replay_schema(tmp_path) -> None:
    row = _row(condition=smoke.BASE_MATCHED_CONDITION, example_id="eval", scores=[0.8, 0.2, 0.1, 0.0])
    state = smoke._fit_receiver([row], ridge=0.1, matched_weight=1.0, control_weight=1.0)
    converted = smoke._convert_base_row(
        base_row=row,
        display_condition=smoke.MATCHED_CONDITION,
        example=_example(),
        state=state,
        latency_ms=0.0,
    )
    path = tmp_path / "predictions.jsonl"
    smoke._write_jsonl(path, [converted])
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["condition"] == smoke.MATCHED_CONDITION
    assert "scores" in loaded["metadata"]
    assert loaded["answer_index"] == 0
    assert loaded["prior_index"] == 1
