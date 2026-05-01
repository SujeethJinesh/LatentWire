from __future__ import annotations

from scripts import build_source_private_commonsenseqa_bridge_contract as csqa_contract


def _csqa_row(question: str, answer_key: str = "C") -> dict[str, object]:
    return {
        "id": question.replace(" ", "_"),
        "question": question,
        "question_concept": "hidden dataset concept",
        "choices": {
            "text": ["choice a", "choice b", "choice c", "choice d", "choice e"],
            "label": ["A", "B", "C", "D", "E"],
        },
        "answerKey": answer_key,
    }


def test_canonical_commonsenseqa_row_maps_answer_and_hides_question_concept() -> None:
    row = csqa_contract.canonical_commonsenseqa_row(_csqa_row("Which option?"), source_name="unit", row_index=1)

    assert row["id"] == "Which_option?"
    assert row["question"] == "Which option?"
    assert row["choice_labels"] == ["A", "B", "C", "D", "E"]
    assert row["answer_index"] == 2
    assert row["answer_label"] == "C"
    assert "question_concept" not in row
    assert "answerKey" not in row


def test_commonsenseqa_contract_passes_with_labeled_public_splits(tmp_path, monkeypatch) -> None:
    rows_by_split = {
        "train": [_csqa_row("train question")],
        "validation": [_csqa_row("validation question")],
    }

    def fake_load(split: str, hf_dataset: str, cache_dir) -> list[dict[str, object]]:
        assert hf_dataset == "commonsenseqa_unit"
        return [
            csqa_contract.canonical_commonsenseqa_row(
                row,
                source_name=f"{hf_dataset}/{split}",
                row_index=index + 1,
            )
            for index, row in enumerate(rows_by_split[split])
        ]

    monkeypatch.setattr(csqa_contract, "_load_labeled_split", fake_load)
    monkeypatch.setattr(csqa_contract, "EXPECTED_LABELED_COUNTS", {"train": 1, "validation": 1})

    payload = csqa_contract.build_contract(
        output_dir=tmp_path / "out",
        hf_dataset="commonsenseqa_unit",
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is True
    assert payload["checks"]["labeled_splits_disjoint"] is True
    assert payload["checks"]["question_concept_forbidden_for_this_gate"] is True
    assert payload["checks"]["test_labels_unavailable_in_hf_mirror"] is True
    assert "answerKey" in payload["method_contract"]["forbidden_source_fields"]
    assert "candidate_derangement" in payload["method_contract"]["required_controls"]
    assert (tmp_path / "out" / "official_splits" / "commonsenseqa_validation.jsonl").exists()


def test_commonsenseqa_contract_fails_on_cross_split_overlap(tmp_path, monkeypatch) -> None:
    shared = _csqa_row("duplicate question")
    rows_by_split = {"train": [shared], "validation": [shared]}

    def fake_load(split: str, hf_dataset: str, cache_dir) -> list[dict[str, object]]:
        return [
            csqa_contract.canonical_commonsenseqa_row(
                row,
                source_name=f"{hf_dataset}/{split}",
                row_index=index + 1,
            )
            for index, row in enumerate(rows_by_split[split])
        ]

    monkeypatch.setattr(csqa_contract, "_load_labeled_split", fake_load)
    monkeypatch.setattr(csqa_contract, "EXPECTED_LABELED_COUNTS", {"train": 1, "validation": 1})

    payload = csqa_contract.build_contract(output_dir=tmp_path / "out", run_date="2026-05-01")

    assert payload["pass_gate"] is False
    assert payload["checks"]["labeled_splits_disjoint"] is False
    assert payload["labeled_overlap_matrix"]["train"]["validation"] == 1
