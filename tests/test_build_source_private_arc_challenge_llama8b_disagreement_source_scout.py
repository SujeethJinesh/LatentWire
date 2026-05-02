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


def test_materialize_source_caches_forwards_mps_workaround_controls(tmp_path, monkeypatch) -> None:
    validation_rows = [DummyRow(["alpha", "beta"], answer_index=1)]
    test_rows = [DummyRow(["gamma", "delta"], answer_index=0)]
    captured = {}

    def fake_lm_choice_loglikelihood_scores(rows, **kwargs):
        captured["row_count"] = len(rows)
        captured.update(kwargs)
        return [[0.1, 0.9], [0.8, 0.2]], [1, 0], {"kind": "fake_lm"}

    monkeypatch.setattr(
        scout.arc_gate,
        "_lm_choice_loglikelihood_scores",
        fake_lm_choice_loglikelihood_scores,
    )

    audit = scout._materialize_source_caches(
        validation_rows=validation_rows,
        test_rows=test_rows,
        validation_cache=tmp_path / "validation" / "source_prediction_cache.jsonl",
        test_cache=tmp_path / "test" / "source_prediction_cache.jsonl",
        source_family="dummy",
        source_model=tmp_path / "model",
        source_lm_device="mps",
        source_lm_dtype="float16",
        source_lm_max_length=192,
        source_lm_normalization="mean",
        source_lm_prompt_mode="qa",
        source_lm_attn_implementation="eager",
        source_lm_choice_batch_size=1,
        local_files_only=True,
        force_rematerialize=True,
    )

    assert captured["row_count"] == 2
    assert captured["attn_implementation"] == "eager"
    assert captured["choice_batch_size"] == 1
    assert audit["lm_state"]["kind"] == "fake_lm"
