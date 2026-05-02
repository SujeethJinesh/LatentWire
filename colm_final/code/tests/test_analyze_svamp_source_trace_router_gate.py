import math

from scripts import analyze_svamp_source_trace_router_gate as gate


def test_equations_extract_valid_arithmetic_and_op_counts():
    rows = gate._equations("First 2 + 3 = 5. Then 4*2=8. But 9 - 1 = 3 is wrong.")

    assert len(rows) == 3
    assert [row["op"] for row in rows] == ["add", "mul", "sub"]
    assert [row["valid"] for row in rows] == [True, True, False]


def test_permuted_equation_text_rotates_rhs_results():
    text = "First 2 + 3 = 5. Then 4 * 2 = 8."

    permuted = gate._permuted_equation_text(text, example_id="ex1", seed=1)

    assert permuted != text
    assert gate._equations(permuted)[0]["valid"] is False
    assert gate._equations(permuted)[1]["valid"] is False


def test_feature_values_use_permuted_equations_for_control():
    source_row = {
        "example_id": "ex1",
        "answer": "8",
        "prediction": "We use 2 + 3 = 5. Then 4 * 2 = 8. Answer: 8",
    }
    prompt = "A problem with 2, 3, 4, and 2."

    matched = gate._feature_values(
        source_row=source_row,
        prompt=prompt,
        example_id="ex1",
        permute_equations=False,
        permutation_seed=1,
    )
    permuted = gate._feature_values(
        source_row=source_row,
        prompt=prompt,
        example_id="ex1",
        permute_equations=True,
        permutation_seed=1,
    )

    assert matched["source_equation_valid_fraction"] == 1.0
    assert matched["source_final_value_matches_last_equation"] == 1.0
    assert matched["source_answer_reused_in_trace"] == 1.0
    assert math.isclose(matched["prompt_number_coverage"], 1.0)
    assert permuted["source_equation_valid_fraction"] == 0.0
    assert permuted["source_final_value_matches_last_equation"] == 0.0
    assert permuted["source_answer_reused_in_trace"] == 0.0


def test_fit_stump_prefers_feature_that_helps_without_harm():
    rows = [
        {
            "fold": 0,
            "features": {"matched": {"source_equation_valid_fraction": 1.0}},
            "raw_conditions": {"matched": {"correct": True}},
            "fallback_correct": False,
        },
        {
            "fold": 1,
            "features": {"matched": {"source_equation_valid_fraction": 0.0}},
            "raw_conditions": {"matched": {"correct": False}},
            "fallback_correct": True,
        },
    ]

    rule = gate._fit_stump(rows, train_folds=None, accept_penalty=0.0)

    assert rule["feature"] == "source_equation_valid_fraction"
    assert rule["direction"] == "ge"
    assert 0.0 < rule["threshold"] <= 1.0
    assert rule["train_help"] == 1
    assert rule["train_harm"] == 0
