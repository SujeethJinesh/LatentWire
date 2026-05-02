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


def test_flip_matrix_and_exact_sign_p_value() -> None:
    candidate = [
        _row("a", correct=False),
        _row("b", correct=True),
        _row("c", correct=False),
        _row("d", correct=True),
    ]
    target = [
        _row("a", correct=False),
        _row("b", correct=False),
        _row("c", correct=True),
        _row("d", correct=True),
    ]
    assert diagnostics._flip_matrix(candidate_records=candidate, target_records=target) == {
        "target_wrong_candidate_wrong": 1,
        "target_wrong_candidate_right": 1,
        "target_right_candidate_wrong": 1,
        "target_right_candidate_right": 1,
    }
    assert diagnostics._exact_sign_p_value(help_count=2, harm_count=0) == 0.5


def test_row_validity_catches_order_and_coverage() -> None:
    rows = [
        _row("b", correct=False, prediction=""),
        _row("a", correct=True, prediction="answer is 3"),
    ]
    validity = diagnostics._row_validity(label="candidate", records=rows, reference_ids=["a", "b"])
    assert validity["row_count_matches"] is True
    assert validity["set_id_parity"] is True
    assert validity["ordered_id_parity"] is False
    assert validity["numeric_extraction_coverage"] == 1
    assert validity["empty_predictions"] == 1


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
    assert rows[0]["candidate_same_numeric_as_source"] is True
    assert rows[0]["source_copy_risk"] is True


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
        "flip_matrix": {
            "target_wrong_candidate_wrong": 28,
            "target_wrong_candidate_right": 2,
            "target_right_candidate_wrong": 0,
            "target_right_candidate_right": 2,
        },
        "paired_sign_p_value": 0.5,
        "oracle_bound": {"correct": 4, "left_only": 2, "right_only": 0, "both": 2, "neither": 28},
        "candidate_only_win_support": {
            "n": 2,
            "source_correct": 1,
            "source_wrong": 1,
            "text_correct": 0,
            "text_wrong": 2,
            "candidate_same_numeric_as_source": 1,
            "candidate_same_numeric_as_text": 0,
            "latent_noncopy_help": 1,
            "text_poison": 0,
        },
        "text_to_text_loss_support": {
            "n": 1,
            "source_correct": 0,
            "source_wrong": 1,
            "text_correct": 0,
            "text_wrong": 1,
            "candidate_same_numeric_as_source": 0,
            "candidate_same_numeric_as_text": 0,
            "latent_noncopy_help": 0,
            "text_poison": 1,
        },
        "validity": [
            {
                "label": "candidate",
                "row_count": 32,
                "expected_count": 32,
                "ordered_id_parity": True,
                "numeric_extraction_coverage": 32,
                "empty_predictions": 0,
            }
        ],
        "diagnostic_gate": {
            "status": "positive_noncopy_but_oracle_saturated",
            "validity_ok": True,
            "source_copy_rate_on_candidate_only_wins": 0.5,
            "oracle_headroom_examples": 0,
        },
        "candidate_only_wins": [
            {
                "index": 3,
                "example_id": "abc",
                "source_correct": True,
                "text_correct": False,
                "target_correct": False,
                "candidate_correct": True,
                "candidate_same_numeric_as_source": True,
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
    assert "## Paired Flip Matrix" in text
    assert "status: `positive_noncopy_but_oracle_saturated`" in text
    assert "| 3 | abc | T | F | F | T | T | '42' | prompt |" in text


def test_payload_round_trip_json() -> None:
    payload = {"summary_metrics": {"candidate_accuracy": 0.125}, "paired_vs_target": {"win": 2}}
    dumped = json.dumps(payload, sort_keys=True)
    assert json.loads(dumped)["summary_metrics"]["candidate_accuracy"] == 0.125
