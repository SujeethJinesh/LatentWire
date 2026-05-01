from __future__ import annotations

import json

from scripts import build_source_private_hellaswag_control_suite as controls
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _row(row_id: str, question: str, answer_index: int) -> dict[str, object]:
    choices = [
        f"{question} ending A",
        f"{question} ending B",
        f"{question} ending C",
        f"{question} ending D",
    ]
    labels = ["A", "B", "C", "D"]
    return {
        "id": row_id,
        "content_id": row_id,
        "question": question,
        "choices": choices,
        "choice_labels": labels,
        "answer_index": answer_index,
        "answer_label": labels[answer_index],
        "source_name": "unit",
    }


def _write_jsonl(path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_hellaswag_control_suite_materializes_label_copy_and_metadata_controls(tmp_path, monkeypatch) -> None:
    train_rows = [
        _row("train-one", "train one", 0),
        _row("train-two", "train two", 1),
        _row("train-three", "train three", 1),
    ]
    eval_rows = [
        _row("eval-one", "eval one", 0),
        _row("eval-two", "eval two", 2),
        _row("eval-three", "eval three", 1),
    ]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(eval_path, eval_rows)

    source_indices = [0, 1, 1]
    anchor_path = tmp_path / "anchor_predictions.jsonl"
    anchor_rows = []
    for raw, source_index in zip(eval_rows, source_indices, strict=True):
        anchor_rows.append(
            {
                "condition": arc_gate.MATCHED_CONDITION,
                "row_id": raw["id"],
                "content_id": raw["content_id"],
                "metadata": {
                    "source_selected_index": source_index,
                    "source_selected_label": raw["choice_labels"][source_index],
                    "source_selected_choice_sha256": arc_gate._sha256_text(raw["choices"][source_index]),
                },
            }
        )
    _write_jsonl(anchor_path, anchor_rows)

    fixed_result = tmp_path / "fixed.json"
    fixed_result.write_text(
        json.dumps(
            {
                "budget_bytes": 2,
                "feature_dim": 16,
                "code_dim": 8,
                "seed": 7,
                "feature_mode": "hashed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_metadata(*, split: str, hf_dataset: str, hf_cache_dir):
        rows = train_rows if split == "train" else eval_rows
        return {
            str(row["content_id"]): {
                "activity_label": "unit_activity" if index < 2 else "rare_activity",
                "split_type": "indomain" if index % 2 == 0 else "zeroshot",
            }
            for index, row in enumerate(rows)
        }

    monkeypatch.setattr(controls, "_load_hf_metadata", fake_metadata)

    payload = controls.build_control_suite(
        output_dir=tmp_path / "out",
        train_path=train_path,
        eval_path=eval_path,
        anchor_predictions=anchor_path,
        fixed_result=fixed_result,
        hf_dataset="unit",
        hf_cache_dir=tmp_path / "cache",
        bootstrap_samples=20,
        run_date="2026-05-01",
    )

    assert "source_label_text_copy" in payload["condition_metrics"]
    assert "same_activity_shuffled_source_packet" in payload["condition_metrics"]
    assert "same_split_type_shuffled_source_packet" in payload["condition_metrics"]
    assert "activity_label_train_majority_prior" in payload["condition_metrics"]
    assert "split_type_train_majority_prior" in payload["condition_metrics"]
    assert "strict_non_label_copy_pass_gate" in payload
    assert payload["condition_metrics"]["source_label_text_copy"]["n"] == len(eval_rows)
    assert (tmp_path / "out" / "hellaswag_control_suite.json").exists()
    assert (tmp_path / "out" / "predictions.jsonl").exists()
