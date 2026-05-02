from __future__ import annotations

from scripts import materialize_masked_process_verifier_sidecars as process_sidecar


def test_mask_answer_values_replaces_exact_numeric_answer() -> None:
    text = "14 - 7 = 7, so the answer is 7 cups."

    masked = process_sidecar.mask_answer_values(text, {"7"})

    assert "14" in masked
    assert " 7" not in masked
    assert masked.count("<ANS>") == 3


def test_process_features_exclude_masked_answer_numbers() -> None:
    features = process_sidecar.process_features(
        "Mary needs 14 - 7 = 7 more cups.",
        answer_values={"7"},
    )

    assert "num:14" in features
    assert "num:7" not in features
    assert "op_sub" in features


def test_score_candidate_prefers_process_overlap() -> None:
    source = {"op_sub", "num:14", "w:flour"}
    candidate_good = {"op_sub", "num:14", "w:flour"}
    candidate_bad = {"op_add", "num:60", "w:sugar"}

    assert process_sidecar._score_candidate(source, candidate_good) > process_sidecar._score_candidate(source, candidate_bad)
