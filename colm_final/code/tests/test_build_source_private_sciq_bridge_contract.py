from __future__ import annotations

from scripts import build_source_private_sciq_bridge_contract as sciq_contract


def _sciq_row(question: str, correct: str = "answer") -> dict[str, str]:
    return {
        "question": question,
        "correct_answer": correct,
        "distractor1": "wrong one",
        "distractor2": "wrong two",
        "distractor3": "wrong three",
        "support": "The answer is not visible to the packet gate.",
    }


def test_canonical_sciq_row_shuffles_choices_and_hides_support() -> None:
    row = sciq_contract.canonical_sciq_row(_sciq_row("What is tested?"), source_name="unit", row_index=1)

    assert row["id"].startswith("sciq_")
    assert row["question"] == "What is tested?"
    assert row["choice_labels"] == ["A", "B", "C", "D"]
    assert row["choices"][row["answer_index"]] == "answer"
    assert row["answer_label"] in row["choice_labels"]
    assert "support" not in row
    assert "correct_answer" not in row


def test_sciq_contract_passes_with_materialized_public_splits(tmp_path, monkeypatch) -> None:
    rows_by_split = {
        "train": [_sciq_row("train question")],
        "validation": [_sciq_row("validation question")],
        "test": [_sciq_row("test question")],
    }

    def fake_load(split: str, hf_dataset: str, cache_dir) -> list[dict[str, str]]:
        assert hf_dataset == "sciq_unit"
        return [
            sciq_contract.canonical_sciq_row(row, source_name=f"{hf_dataset}/{split}", row_index=index + 1)
            for index, row in enumerate(rows_by_split[split])
        ]

    monkeypatch.setattr(sciq_contract, "_load_official_split", fake_load)
    monkeypatch.setattr(sciq_contract, "EXPECTED_OFFICIAL_COUNTS", {"train": 1, "validation": 1, "test": 1})

    payload = sciq_contract.build_contract(
        output_dir=tmp_path / "out",
        hf_dataset="sciq_unit",
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is True
    assert payload["checks"]["official_splits_disjoint"] is True
    assert payload["checks"]["support_context_forbidden_for_this_gate"] is True
    assert "correct_answer" in payload["method_contract"]["forbidden_source_fields"]
    assert "candidate_derangement" in payload["method_contract"]["required_controls"]
    assert (tmp_path / "out" / "official_splits" / "sciq_test.jsonl").exists()


def test_sciq_contract_fails_on_cross_split_overlap(tmp_path, monkeypatch) -> None:
    shared = _sciq_row("duplicate question")
    rows_by_split = {"train": [shared], "validation": [shared], "test": [_sciq_row("test question")]}

    def fake_load(split: str, hf_dataset: str, cache_dir) -> list[dict[str, str]]:
        return [
            sciq_contract.canonical_sciq_row(row, source_name=f"{hf_dataset}/{split}", row_index=index + 1)
            for index, row in enumerate(rows_by_split[split])
        ]

    monkeypatch.setattr(sciq_contract, "_load_official_split", fake_load)
    monkeypatch.setattr(sciq_contract, "EXPECTED_OFFICIAL_COUNTS", {"train": 1, "validation": 1, "test": 1})

    payload = sciq_contract.build_contract(output_dir=tmp_path / "out", run_date="2026-05-01")

    assert payload["pass_gate"] is False
    assert payload["checks"]["official_splits_disjoint"] is False
    assert payload["official_overlap_matrix"]["train"]["validation"] == 1
