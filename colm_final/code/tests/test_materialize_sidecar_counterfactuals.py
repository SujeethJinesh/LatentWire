from __future__ import annotations

from scripts import materialize_sidecar_counterfactuals as counterfactuals


def test_source_answer_masked_zeroes_final_and_verified_values() -> None:
    scores = [
        {"label": "gold", "value": "7", "score": 6.0},
        {"label": "verified", "value": "9", "score": 5.0},
        {"label": "other", "value": "2", "score": 1.0},
    ]
    profile = {"final": "7", "verified": {"9"}}

    transformed = counterfactuals.transform_scores(scores, profile=profile, mode="source_answer_masked")

    by_value = {item["value"]: item["score"] for item in transformed}
    assert by_value == {"7": 0.0, "9": 0.0, "2": 1.0}
    assert transformed[0]["value"] == "2"


def test_source_final_only_keeps_only_final_match() -> None:
    scores = [
        {"label": "gold", "value": "7", "score": 6.0},
        {"label": "verified", "value": "9", "score": 5.0},
        {"label": "other", "value": "2", "score": 1.0},
    ]
    profile = {"final": "7", "verified": {"9"}}

    transformed = counterfactuals.transform_scores(scores, profile=profile, mode="source_final_only")

    by_value = {item["value"]: item["score"] for item in transformed}
    assert by_value == {"7": 3.0, "9": 0.0, "2": 0.0}
    assert transformed[0]["value"] == "7"
