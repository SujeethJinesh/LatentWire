from __future__ import annotations

from scripts import analyze_answer_null_predicate_syndrome as syndrome


def test_predicate_syndrome_masks_answer_values() -> None:
    predicates = syndrome.predicate_syndrome(
        "Mary needs 14 - 7 = 7 more cups of flour.",
        answer_values={"7"},
    )

    assert "op_sub" in predicates
    assert "unit:cups" in predicates
    assert "unit:flour" in predicates
    assert all("7" not in item for item in predicates)


def test_score_candidate_weights_operation_and_relation_overlap() -> None:
    predicates = {"op_sub", "rel_difference", "unit:cups"}
    good = {"op_sub", "rel_difference", "unit:cups"}
    bad = {"op_add", "rel_total", "unit:cups"}

    assert syndrome._score_candidate(predicates, good) > syndrome._score_candidate(predicates, bad)
