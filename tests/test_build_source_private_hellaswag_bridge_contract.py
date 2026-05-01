from __future__ import annotations

from scripts import build_source_private_hellaswag_bridge_contract as hs_contract


def _hellaswag_row(context: str, label: str = "2") -> dict[str, object]:
    return {
        "ind": abs(hash(context)) % 100000,
        "activity_label": "metadata should not be source-visible",
        "ctx_a": context,
        "ctx_b": "then",
        "ctx": context,
        "endings": [
            "falls into a bright river.",
            "talks to the crowd and waves.",
            "continues the same task carefully.",
            "turns into a cartoon spaceship.",
        ],
        "source_id": f"activitynet~{context}",
        "split": "train",
        "split_type": "indomain",
        "label": label,
    }


def test_canonical_hellaswag_row_maps_label_and_hides_metadata() -> None:
    row = hs_contract.canonical_hellaswag_row(_hellaswag_row("A person cooks dinner."), source_name="unit", row_index=1)

    assert row["question"] == "A person cooks dinner."
    assert row["choice_labels"] == ["A", "B", "C", "D"]
    assert row["answer_index"] == 2
    assert row["answer_label"] == "C"
    assert "activity_label" not in row
    assert "source_id" not in row
    assert "label" not in row


def test_hellaswag_contract_passes_with_materialized_labeled_splits(tmp_path, monkeypatch) -> None:
    rows_by_split = {
        "train": [_hellaswag_row("train context")],
        "validation": [_hellaswag_row("validation context")],
    }

    def fake_load(split: str, hf_dataset: str, cache_dir) -> list[dict[str, object]]:
        assert hf_dataset == "hellaswag_unit"
        return [
            hs_contract.canonical_hellaswag_row(
                row,
                source_name=f"{hf_dataset}/{split}",
                row_index=index + 1,
            )
            for index, row in enumerate(rows_by_split[split])
        ]

    monkeypatch.setattr(hs_contract, "_load_labeled_split", fake_load)
    monkeypatch.setattr(hs_contract, "EXPECTED_LABELED_COUNTS", {"train": 1, "validation": 1})

    payload = hs_contract.build_contract(
        output_dir=tmp_path / "out",
        hf_dataset="hellaswag_unit",
        validation_slice_rows=1,
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is True
    assert payload["checks"]["labeled_splits_disjoint"] is True
    assert payload["checks"]["metadata_fields_forbidden"] is True
    assert payload["checks"]["official_test_labels_public"] is False
    assert "activity_label" in payload["method_contract"]["forbidden_source_fields"]
    assert "candidate_derangement" in payload["method_contract"]["required_controls"]
    assert (tmp_path / "out" / "official_splits" / "hellaswag_validation_first1.jsonl").exists()


def test_hellaswag_contract_fails_on_cross_split_overlap(tmp_path, monkeypatch) -> None:
    shared = _hellaswag_row("duplicate context")
    rows_by_split = {"train": [shared], "validation": [shared]}

    def fake_load(split: str, hf_dataset: str, cache_dir) -> list[dict[str, object]]:
        return [
            hs_contract.canonical_hellaswag_row(
                row,
                source_name=f"{hf_dataset}/{split}",
                row_index=index + 1,
            )
            for index, row in enumerate(rows_by_split[split])
        ]

    monkeypatch.setattr(hs_contract, "_load_labeled_split", fake_load)
    monkeypatch.setattr(hs_contract, "EXPECTED_LABELED_COUNTS", {"train": 1, "validation": 1})

    payload = hs_contract.build_contract(output_dir=tmp_path / "out", validation_slice_rows=1, run_date="2026-05-01")

    assert payload["pass_gate"] is False
    assert payload["checks"]["labeled_splits_disjoint"] is False
    assert payload["labeled_overlap_matrix"]["train"]["validation"] == 1
