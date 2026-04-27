import json

import pytest

from scripts import analyze_condition_likelihood_receiver_gate as gate


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _score_row(example_id, *, target_correct, source_correct, top_label="source", margin=2.0):
    target_score = 0.0
    source_score = target_score + margin if top_label == "source" else target_score - margin
    return {
        "example_id": example_id,
        "candidate_scores": [
            {
                "label": "target",
                "score": target_score,
                "candidate_raw_text": "8" if target_correct else "0",
                "candidate_correct": target_correct,
            },
            {
                "label": "source",
                "score": source_score,
                "candidate_raw_text": "8" if source_correct else "1",
                "candidate_correct": source_correct,
            },
        ],
    }


def test_condition_specific_controls_use_their_own_candidate_correctness(tmp_path):
    matched = tmp_path / "matched.jsonl"
    shuffled = tmp_path / "shuffled.jsonl"
    target_only = tmp_path / "target_only.jsonl"
    target_set = tmp_path / "target_set.json"
    _write_jsonl(
        matched,
        [
            _score_row("clean", target_correct=False, source_correct=True, margin=3.0),
            _score_row("preserve", target_correct=True, source_correct=False, margin=-1.0),
        ],
    )
    _write_jsonl(
        shuffled,
        [
            _score_row("clean", target_correct=False, source_correct=False, margin=3.0),
            _score_row("preserve", target_correct=True, source_correct=False, margin=3.0),
        ],
    )
    _write_jsonl(
        target_only,
        [
            _score_row("clean", target_correct=False, source_correct=False, top_label="target", margin=3.0),
            _score_row("preserve", target_correct=True, source_correct=False, top_label="target", margin=3.0),
        ],
    )
    target_set.write_text(
        json.dumps(
            {
                "ids": {
                    "clean_residual_targets": ["clean"],
                    "clean_source_only": ["clean"],
                    "target_self_repair": ["preserve"],
                }
            }
        ),
        encoding="utf-8",
    )
    rows, ids, conditions = gate._prepare_rows(
        paths={"matched": matched, "shuffled_source": shuffled, "target_only": target_only},
        target_set_path=target_set,
        fallback_label="target",
        outer_folds=2,
        max_sidecar_bits=8,
    )

    result = gate._evaluate(
        rows=rows,
        target_ids=ids,
        conditions=conditions,
        global_rule={"feature": "margin", "threshold": 1.0, "direction": "ge"},
    )

    assert result["condition_summaries"]["matched"]["correct_count"] == 2
    assert result["condition_summaries"]["shuffled_source"]["correct_count"] == 0
    assert result["condition_summaries"]["target_only"]["correct_count"] == 1
    assert result["source_necessary_clean_ids"] == ["clean"]
    assert result["accepted_harm"] == 0


def test_control_clean_union_removes_recovered_control_ids(tmp_path):
    matched = tmp_path / "matched.jsonl"
    shuffled = tmp_path / "shuffled.jsonl"
    target_set = tmp_path / "target_set.json"
    _write_jsonl(matched, [_score_row("clean", target_correct=False, source_correct=True, margin=3.0)])
    _write_jsonl(shuffled, [_score_row("clean", target_correct=False, source_correct=True, margin=3.0)])
    target_set.write_text(
        json.dumps({"ids": {"clean_residual_targets": ["clean"], "clean_source_only": ["clean"], "target_self_repair": []}}),
        encoding="utf-8",
    )
    rows, ids, conditions = gate._prepare_rows(
        paths={"matched": matched, "shuffled_source": shuffled},
        target_set_path=target_set,
        fallback_label="target",
        outer_folds=2,
        max_sidecar_bits=8,
    )

    result = gate._evaluate(
        rows=rows,
        target_ids=ids,
        conditions=conditions,
        global_rule={"feature": "margin", "threshold": 1.0, "direction": "ge"},
    )

    assert result["control_clean_union_ids"] == ["clean"]
    assert result["source_necessary_clean_ids"] == []


def test_duplicate_answer_not_counted_source_necessary(tmp_path):
    matched = tmp_path / "matched.jsonl"
    target_set = tmp_path / "target_set.json"
    _write_jsonl(
        matched,
        [
            {
                "example_id": "clean",
                "candidate_scores": [
                    {"label": "target", "score": 0.0, "candidate_raw_text": "5", "candidate_correct": False},
                    {"label": "source", "score": 3.0, "candidate_raw_text": "5", "candidate_correct": True},
                ],
            }
        ],
    )
    target_set.write_text(
        json.dumps({"ids": {"clean_residual_targets": ["clean"], "clean_source_only": ["clean"], "target_self_repair": []}}),
        encoding="utf-8",
    )
    rows, ids, conditions = gate._prepare_rows(
        paths={"matched": matched},
        target_set_path=target_set,
        fallback_label="target",
        outer_folds=2,
        max_sidecar_bits=8,
    )

    result = gate._evaluate(
        rows=rows,
        target_ids=ids,
        conditions=conditions,
        global_rule={"feature": "margin", "threshold": 1.0, "direction": "ge"},
    )

    assert result["condition_summaries"]["matched"]["duplicate_answer_ids"] == ["clean"]
    assert result["duplicate_answer_clean_ids"] == ["clean"]
    assert result["source_necessary_clean_ids"] == []


def test_condition_sketch_ids_must_match(tmp_path):
    matched = tmp_path / "matched.jsonl"
    shuffled = tmp_path / "shuffled.jsonl"
    _write_jsonl(matched, [_score_row("a", target_correct=False, source_correct=True)])
    _write_jsonl(shuffled, [_score_row("b", target_correct=False, source_correct=True)])

    with pytest.raises(ValueError, match="does not match matched IDs"):
        gate._load_condition_sketches(
            {"matched": matched, "shuffled_source": shuffled},
            max_sidecar_bits=8,
        )
