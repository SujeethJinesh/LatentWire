from __future__ import annotations

from scripts import build_source_private_openbookqa_bridge_contract as obqa_contract


def _obqa_row(question: str, answer_key: str = "B") -> dict[str, object]:
    return {
        "id": question.replace(" ", "_"),
        "question_stem": question,
        "choices": {"text": ["choice a", "choice b", "choice c", "choice d"], "label": ["A", "B", "C", "D"]},
        "answerKey": answer_key,
        "fact1": "This open-book fact is not visible in this gate.",
    }


def test_canonical_openbookqa_row_maps_answer_and_hides_fact() -> None:
    row = obqa_contract.canonical_openbookqa_row(_obqa_row("Which option?"), source_name="unit", row_index=1)

    assert row["id"] == "Which_option?"
    assert row["question"] == "Which option?"
    assert row["choice_labels"] == ["A", "B", "C", "D"]
    assert row["answer_index"] == 1
    assert row["answer_label"] == "B"
    assert "fact1" not in row
    assert "answerKey" not in row


def test_openbookqa_contract_passes_with_materialized_public_splits(tmp_path, monkeypatch) -> None:
    rows_by_split = {
        "train": [_obqa_row("train question")],
        "validation": [_obqa_row("validation question")],
        "test": [_obqa_row("test question")],
    }

    def fake_load(split: str, hf_dataset: str, hf_config: str, cache_dir) -> list[dict[str, object]]:
        assert hf_dataset == "openbookqa_unit"
        assert hf_config == "main"
        return [
            obqa_contract.canonical_openbookqa_row(
                row,
                source_name=f"{hf_dataset}/{hf_config}/{split}",
                row_index=index + 1,
            )
            for index, row in enumerate(rows_by_split[split])
        ]

    monkeypatch.setattr(obqa_contract, "_load_official_split", fake_load)
    monkeypatch.setattr(obqa_contract, "EXPECTED_OFFICIAL_COUNTS", {"train": 1, "validation": 1, "test": 1})

    payload = obqa_contract.build_contract(
        output_dir=tmp_path / "out",
        hf_dataset="openbookqa_unit",
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is True
    assert payload["checks"]["official_splits_disjoint"] is True
    assert payload["checks"]["openbook_fact_forbidden_for_this_gate"] is True
    assert "answerKey" in payload["method_contract"]["forbidden_source_fields"]
    assert "candidate_derangement" in payload["method_contract"]["required_controls"]
    assert (tmp_path / "out" / "official_splits" / "openbookqa_test.jsonl").exists()


def test_openbookqa_contract_fails_on_cross_split_overlap(tmp_path, monkeypatch) -> None:
    shared = _obqa_row("duplicate question")
    rows_by_split = {"train": [shared], "validation": [shared], "test": [_obqa_row("test question")]}

    def fake_load(split: str, hf_dataset: str, hf_config: str, cache_dir) -> list[dict[str, object]]:
        return [
            obqa_contract.canonical_openbookqa_row(
                row,
                source_name=f"{hf_dataset}/{hf_config}/{split}",
                row_index=index + 1,
            )
            for index, row in enumerate(rows_by_split[split])
        ]

    monkeypatch.setattr(obqa_contract, "_load_official_split", fake_load)
    monkeypatch.setattr(obqa_contract, "EXPECTED_OFFICIAL_COUNTS", {"train": 1, "validation": 1, "test": 1})

    payload = obqa_contract.build_contract(output_dir=tmp_path / "out", run_date="2026-05-01")

    assert payload["pass_gate"] is False
    assert payload["checks"]["official_splits_disjoint"] is False
    assert payload["official_overlap_matrix"]["train"]["validation"] == 1
