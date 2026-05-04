from __future__ import annotations

import json

from scripts import build_source_private_arc_challenge_soft_prefix_resonance_gate as gate
from scripts import run_source_private_arc_challenge_fixed_packet_gate as arc_gate


def _row(row_id: str, content_id: str) -> arc_gate.ArcRow:
    return arc_gate.ArcRow(
        row_id=row_id,
        content_id=content_id,
        question=f"Question {row_id}?",
        choices=("A choice", "B choice"),
        choice_labels=("A", "B"),
        answer_index=0,
        answer_label="A",
    )


def test_disagreement_reader_filters_split_and_agreement(tmp_path) -> None:
    path = tmp_path / "agreement.csv"
    path.write_text(
        "split,row_id,content_id,alt_source_selected_index,qwen_source_selected_index,agree,answer_index,"
        "alt_source_correct,qwen_source_correct\n"
        "validation,r0,c0,0,1,False,0,True,False\n"
        "validation,r1,c1,0,0,True,0,True,True\n"
        "test,r2,c2,1,0,False,0,False,True\n",
        encoding="utf-8",
    )

    assert gate._read_disagreement_content_ids(agreement_path=path, split="validation", limit=1) == ["c0"]
    assert gate._read_disagreement_content_ids(agreement_path=path, split="test", limit=1) == ["c2"]


def test_combined_rows_and_cache_selection_are_ordered(tmp_path) -> None:
    rows = [_row("r0", "c0"), _row("r1", "c1")]
    row_payloads = [gate._arc_row_payload(row) for row in rows]

    assert row_payloads[0]["id"] == "r0"
    assert row_payloads[0]["choices"] == ["A choice", "B choice"]

    cache_path = tmp_path / "scores.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "content_id": "c1",
                "source_selected_index": 1,
                "source_scores": [0.0, 1.0],
                "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            }
        )
        + "\n"
        + json.dumps(
            {
                "content_id": "c0",
                "source_selected_index": 0,
                "source_scores": [1.0, 0.0],
                "forbidden_source_fields": list(arc_gate.FORBIDDEN_SOURCE_KEYS),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    selected = gate._select_cache_rows(cache_path=cache_path, rows=rows)

    assert [row["content_id"] for row in selected] == ["c0", "c1"]
    assert selected[0]["source_selected_index"] == 0
