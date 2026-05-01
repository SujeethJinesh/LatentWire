from __future__ import annotations

import json

from scripts import build_source_private_arc_challenge_bridge_contract as arc_contract


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _row(question: str, answer: int) -> dict[str, object]:
    return {
        "question": question,
        "choices": ["alpha", "beta", "gamma", "delta"],
        "answer": answer,
    }


def test_contract_passes_for_disjoint_local_arc_slices(tmp_path) -> None:
    validation = tmp_path / "gate.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    combined = tmp_path / "combined.jsonl"
    _write_jsonl(validation, [_row("q1", 0), _row("q2", 1)])
    _write_jsonl(eval_path, [_row("q3", 2)])
    _write_jsonl(combined, [_row("q1", 0), _row("q2", 1), _row("q3", 2)])

    payload = arc_contract.build_contract(
        output_dir=tmp_path / "out",
        validation_smoke_jsonl=validation,
        eval_smoke_jsonl=eval_path,
        combined_smoke_jsonl=combined,
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is True
    assert payload["public_benchmark_result_ready"] is False
    assert payload["checks"]["local_validation_eval_disjoint"] is True
    assert payload["checks"]["combined_smoke_covers_validation_and_eval"] is True
    assert "answerKey" in payload["method_contract"]["forbidden_source_fields"]
    assert "shuffled_source_packet" in payload["method_contract"]["required_controls"]
    assert (tmp_path / "out" / "arc_challenge_bridge_contract.json").exists()
    assert (tmp_path / "out" / "arc_challenge_bridge_contract.md").exists()


def test_contract_fails_when_validation_and_eval_overlap(tmp_path) -> None:
    validation = tmp_path / "gate.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    combined = tmp_path / "combined.jsonl"
    _write_jsonl(validation, [_row("q1", 0), _row("q2", 1)])
    _write_jsonl(eval_path, [_row("q2", 1)])
    _write_jsonl(combined, [_row("q1", 0), _row("q2", 1)])

    payload = arc_contract.build_contract(
        output_dir=tmp_path / "out",
        validation_smoke_jsonl=validation,
        eval_smoke_jsonl=eval_path,
        combined_smoke_jsonl=combined,
        run_date="2026-05-01",
    )

    assert payload["pass_gate"] is False
    assert payload["checks"]["local_validation_eval_disjoint"] is False
    assert payload["local_overlap_matrix"]["validation_smoke"]["evaluation_smoke"] == 1


def test_canonical_arc_row_handles_hf_choice_schema() -> None:
    row = arc_contract.canonical_arc_row(
        {
            "id": "Mercury_SC_407695",
            "question": "Which choice?",
            "choices": {"text": ["one", "two", "three", "four"], "label": ["A", "B", "C", "D"]},
            "answerKey": "D",
        },
        source_name="hf",
        row_index=1,
    )

    assert row["id"] == "Mercury_SC_407695"
    assert row["answer_index"] == 3
    assert row["answer_label"] == "D"
    assert row["content_id"]
