from __future__ import annotations

import json

from scripts import analyze_gsm8k_contract_diagnostics as diagnostics


def _row(example_id: str, *, correct: bool, prediction: str = "x") -> dict[str, object]:
    return {
        "example_id": example_id,
        "correct": correct,
        "prediction": prediction,
    }


def test_oracle_counts() -> None:
    left = [
        _row("a", correct=True),
        _row("b", correct=False),
        _row("c", correct=True),
        _row("d", correct=False),
    ]
    right = [
        _row("a", correct=True),
        _row("b", correct=True),
        _row("c", correct=False),
        _row("d", correct=False),
    ]
    counts = diagnostics._oracle_counts(left, right)
    assert counts == {
        "correct": 3,
        "left_only": 1,
        "right_only": 1,
        "both": 1,
        "neither": 1,
    }


def test_detail_rows_select_candidate_only_wins() -> None:
    examples = [
        {"prompt": "prompt 0", "answers": ["1"]},
        {"prompt": "prompt 1", "answers": ["2"]},
    ]
    candidate = [_row("a", correct=True, prediction="1"), _row("b", correct=False, prediction="3")]
    target = [_row("a", correct=False, prediction="7"), _row("b", correct=True, prediction="2")]
    source = [_row("a", correct=True, prediction="1"), _row("b", correct=False, prediction="3")]
    text = [_row("a", correct=False, prediction="7"), _row("b", correct=False, prediction="3")]
    rows = diagnostics._detail_rows(
        examples=examples,
        candidate_records=candidate,
        target_records=target,
        source_records=source,
        text_records=text,
        selector="candidate_only_wins",
    )
    assert len(rows) == 1
    assert rows[0]["example_id"] == "a"
    assert rows[0]["source_correct"] is True
    assert rows[0]["target_correct"] is False
    assert rows[0]["candidate_correct"] is True


def test_write_markdown_renders_core_sections(tmp_path) -> None:
    payload = {
        "date": "2026-04-22",
        "candidate_label": "dynalign_module_replace_residrank16",
        "config": {
            "source_model": "src",
            "target_model": "tgt",
            "eval_file": "data/gsm8k_eval_70.jsonl",
            "slice_size": 32,
        },
        "summary_metrics": {
            "source_alone_accuracy": 0.03125,
            "source_alone_correct": 1,
            "target_alone_accuracy": 0.0625,
            "target_alone_correct": 2,
            "text_to_text_accuracy": 0.03125,
            "text_to_text_correct": 1,
            "candidate_accuracy": 0.125,
            "candidate_correct": 4,
            "oracle_accuracy": 0.125,
        },
        "paired_vs_target": {"win": 2, "loss": 0, "tie": 30},
        "oracle_bound": {"correct": 4, "left_only": 2, "right_only": 0, "both": 2, "neither": 28},
        "candidate_only_win_support": {"n": 2, "source_correct": 1, "source_wrong": 1, "text_correct": 0, "text_wrong": 2},
        "text_to_text_loss_support": {"n": 1, "source_correct": 0, "source_wrong": 1, "text_correct": 0, "text_wrong": 1},
        "candidate_only_wins": [
            {
                "index": 3,
                "example_id": "abc",
                "source_correct": True,
                "text_correct": False,
                "target_correct": False,
                "candidate_correct": True,
                "reference_answer": "42",
                "prompt_excerpt": "prompt",
            }
        ],
        "candidate_only_losses": [],
        "text_to_text_target_only_losses": [],
    }
    path = tmp_path / "out.md"
    diagnostics._write_markdown(path, payload)
    text = path.read_text()
    assert "# GSM8K Contract Diagnostics" in text
    assert "oracle(target, candidate)" in text
    assert "source correctness on candidate-only wins: `1 / 2`" in text
    assert "| 3 | abc | T | F | F | T | '42' | prompt |" in text


def test_payload_round_trip_json() -> None:
    payload = {"summary_metrics": {"candidate_accuracy": 0.125}, "paired_vs_target": {"win": 2}}
    dumped = json.dumps(payload, sort_keys=True)
    assert json.loads(dumped)["summary_metrics"]["candidate_accuracy"] == 0.125
