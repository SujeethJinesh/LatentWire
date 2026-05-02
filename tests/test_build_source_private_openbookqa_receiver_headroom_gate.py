from __future__ import annotations

import json

from scripts import build_source_private_openbookqa_receiver_headroom_gate as gate


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _row(row_id: str, answer_index: int) -> dict:
    choices = [f"{row_id} choice {index}" for index in range(4)]
    return {
        "id": row_id,
        "content_id": f"content-{row_id}",
        "question": f"What is true for {row_id}?",
        "choices": choices,
        "choice_labels": ["A", "B", "C", "D"],
        "answer_index": answer_index,
        "answer_label": ["A", "B", "C", "D"][answer_index],
    }


def _cache_row(row: dict, source_index: int) -> dict:
    return {
        "row_id": row["id"],
        "content_id": row["content_id"],
        "source_selected_index": source_index,
        "source_selected_choice_sha256": "",
        "source_visible_fields": ["question", "choices"],
        "forbidden_source_fields": ["answer", "answerKey", "answer_index", "answer_label", "gold"],
    }


def test_openbookqa_receiver_gate_writes_artifacts(tmp_path) -> None:
    train_rows = [_row(f"train-{index}", index % 4) for index in range(12)]
    validation_rows = [_row(f"validation-{index}", (index + 1) % 4) for index in range(8)]
    test_rows = [_row(f"test-{index}", (index + 2) % 4) for index in range(8)]
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    validation_cache = tmp_path / "validation_cache.jsonl"
    test_cache = tmp_path / "test_cache.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    _write_jsonl(test_path, test_rows)
    _write_jsonl(validation_cache, [_cache_row(row, (index + 1) % 4) for index, row in enumerate(validation_rows)])
    _write_jsonl(test_cache, [_cache_row(row, (index + 2) % 4) for index, row in enumerate(test_rows)])

    payload = gate.build_receiver_gate(
        output_dir=tmp_path / "out",
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        validation_source_cache=validation_cache,
        test_source_cache=test_cache,
        seeds=[3],
        budget_bytes=2,
        packet_feature_dim=16,
        code_dim=8,
        target_feature_dim=16,
        target_ridge=1.0,
        selector_ridges=[0.1, 1.0],
        threshold_percentiles=[0, 50, 100],
        bootstrap_samples=50,
        min_receiver_lift=0.0,
        min_control_gap=-1.0,
    )

    assert payload["gate"] == "source_private_openbookqa_receiver_headroom_gate"
    assert payload["test_rows"] == 8
    assert "source_label_copy" in payload["per_seed"][0]["condition_metrics"]
    assert "target_public_ridge" in payload["per_seed"][0]["condition_metrics"]
    assert (tmp_path / "out" / "openbookqa_receiver_headroom_gate.json").exists()
    assert (tmp_path / "out" / "receiver_predictions.jsonl").exists()
