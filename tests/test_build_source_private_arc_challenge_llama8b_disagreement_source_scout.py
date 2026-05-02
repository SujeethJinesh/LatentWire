from scripts import build_source_private_arc_challenge_llama8b_disagreement_source_scout as scout


class DummyRow:
    def __init__(self, choices: list[str], answer_index: int = 0) -> None:
        self.row_id = f"row-{len(choices)}-{answer_index}"
        self.content_id = f"content-{len(choices)}-{answer_index}"
        self.question = "Which answer is best?"
        self.choices = choices
        self.answer_index = answer_index


def test_roll_predictions_wraps_per_row_choice_count() -> None:
    rows = [DummyRow(["a", "b", "c"]), DummyRow(["a", "b", "c", "d", "e"])]

    assert scout._roll_predictions(rows, [2, 4]) == [0, 0]


def test_random_predictions_stay_in_row_bounds_and_are_deterministic() -> None:
    rows = [DummyRow(["a", "b", "c"]), DummyRow(["a", "b", "c", "d", "e"])]

    first = scout._random_predictions(rows, seed=17)
    second = scout._random_predictions(rows, seed=17)

    assert first == second
    assert 0 <= first[0] < 3
    assert 0 <= first[1] < 5


def test_source_cache_rows_include_source_private_contract() -> None:
    rows = [DummyRow(["alpha", "beta"], answer_index=1)]

    [cache_row] = scout._source_cache_rows(
        rows=rows,
        predictions=[1],
        source_family="dummy",
        source_model=scout.DEFAULT_LLAMA_MODEL,
        source_lm_prompt_mode="qa",
        source_lm_normalization="mean",
    )

    assert cache_row["source_selected_index"] == 1
    assert cache_row["source_visible_fields"] == ["question", "choices"]
    assert set(cache_row["forbidden_source_fields"]) >= {"answer_index", "answerKey"}
    assert "source_selected_choice_sha256" in cache_row
