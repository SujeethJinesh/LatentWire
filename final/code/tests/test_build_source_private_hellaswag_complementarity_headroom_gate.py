from __future__ import annotations

import json
import pathlib

import pytest

from scripts import build_source_private_hellaswag_complementarity_headroom_gate as gate


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_complementarity_gate_passes_with_stable_oracle_lift(tmp_path: pathlib.Path) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target.jsonl"
    source_rows = []
    target_rows = []
    for index in range(100):
        answer = index % 4
        source_pred = answer if index % 2 == 0 else (answer + 1) % 4
        target_pred = answer if index % 2 == 1 else (answer + 1) % 4
        source_rows.append(
            {
                "answer_index": answer,
                "row_id": str(index),
                "selected_prediction": source_pred,
                "source_label_prediction": source_pred,
            }
        )
        target_rows.append(
            {
                "answer_index": answer,
                "hybrid_vote_on_score_agreement_prediction": target_pred,
                "row_id": str(index),
            }
        )
    _write_jsonl(source_path, source_rows)
    _write_jsonl(target_path, target_rows)

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        source_predictions=source_path,
        target_predictions=target_path,
        bootstrap_samples=100,
    )

    assert payload["pass_gate"] is True
    assert payload["headline"]["source_packet_accuracy"] == 0.5
    assert payload["headline"]["target_side_accuracy"] == 0.5
    assert payload["headline"]["target_or_source_oracle_accuracy"] == 1.0
    assert payload["headline"]["positive_block_count"] == payload["headline"]["block_count"]
    assert (tmp_path / "out" / "hellaswag_complementarity_headroom_gate.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()


def test_row_id_mismatch_fails(tmp_path: pathlib.Path) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target.jsonl"
    _write_jsonl(
        source_path,
        [{"answer_index": 0, "row_id": "a", "selected_prediction": 0, "source_label_prediction": 0}],
    )
    _write_jsonl(
        target_path,
        [{"answer_index": 0, "row_id": "b", "hybrid_vote_on_score_agreement_prediction": 0}],
    )

    with pytest.raises(ValueError, match="row_id order"):
        gate.build_gate(
            output_dir=tmp_path / "out",
            source_predictions=source_path,
            target_predictions=target_path,
        )


def test_gate_fails_when_target_adds_no_headroom(tmp_path: pathlib.Path) -> None:
    source_path = tmp_path / "source.jsonl"
    target_path = tmp_path / "target.jsonl"
    source_rows = []
    target_rows = []
    for index in range(20):
        answer = index % 4
        pred = answer if index % 2 == 0 else (answer + 1) % 4
        source_rows.append(
            {
                "answer_index": answer,
                "row_id": str(index),
                "selected_prediction": pred,
                "source_label_prediction": pred,
            }
        )
        target_rows.append(
            {
                "answer_index": answer,
                "hybrid_vote_on_score_agreement_prediction": pred,
                "row_id": str(index),
            }
        )
    _write_jsonl(source_path, source_rows)
    _write_jsonl(target_path, target_rows)

    payload = gate.build_gate(
        output_dir=tmp_path / "out",
        source_predictions=source_path,
        target_predictions=target_path,
        bootstrap_samples=100,
    )

    assert payload["pass_gate"] is False
    assert payload["headline"]["target_or_source_oracle_lift_vs_source"] == 0.0
