import json

from scripts import analyze_source_confidence_router_gate as gate


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_fit_stump_uses_confidence_feature_to_accept_helpful_source():
    rows = [
        {
            "fold": 0,
            "target_correct": False,
            "gold_answer": ["5"],
            "source_conditions": {
                "matched": {"prediction": "5", "min_top1_prob": 0.7},
            },
        },
        {
            "fold": 1,
            "target_correct": True,
            "gold_answer": ["8"],
            "source_conditions": {
                "matched": {"prediction": "1", "min_top1_prob": 0.1},
            },
        },
    ]

    rule = gate._fit_stump(rows, train_folds=None, accept_penalty=0.0)

    assert rule["feature"] == "min_top1_prob"
    assert rule["direction"] == "ge"
    assert 0.1 < rule["threshold"] <= 0.7
    assert rule["train_help"] == 1
    assert rule["train_harm"] == 0


def test_evaluate_counts_clean_controls_and_source_necessary():
    rows = [
        {
            "index": 0,
            "example_id": "clean",
            "fold": 0,
            "gold_answer": ["5"],
            "target_prediction": "0",
            "target_correct": False,
            "source_conditions": {
                "matched": {"prediction": "5", "min_top1_prob": 0.8},
                "zero_source": None,
                "shuffled_source": {"prediction": "9", "min_top1_prob": 0.8},
                "label_shuffle": {"prediction": "9", "min_top1_prob": 0.8},
                "target_only": None,
            },
        },
        {
            "index": 1,
            "example_id": "preserve",
            "fold": 0,
            "gold_answer": ["8"],
            "target_prediction": "8",
            "target_correct": True,
            "source_conditions": {
                "matched": {"prediction": "1", "min_top1_prob": 0.1},
                "zero_source": None,
                "shuffled_source": {"prediction": "5", "min_top1_prob": 0.8},
                "label_shuffle": {"prediction": "5", "min_top1_prob": 0.8},
                "target_only": None,
            },
        },
    ]
    ids = {
        "clean_residual_targets": {"clean"},
        "target_self_repair": {"preserve"},
        "teacher_only": {"clean"},
    }
    rule = {"feature": "min_top1_prob", "threshold": 0.5, "direction": "ge"}

    result = gate._evaluate(rows=rows, target_ids=ids, global_rule=rule)

    assert result["condition_summaries"]["matched"]["correct_count"] == 2
    assert result["source_necessary_clean_ids"] == ["clean"]
    assert result["control_clean_union_ids"] == []
    assert result["accepted_harm"] == 0


def test_prepare_rows_preserves_reference_order(tmp_path):
    diagnostics = tmp_path / "diag.jsonl"
    target = tmp_path / "target.jsonl"
    target_set = tmp_path / "target_set.json"
    _write_jsonl(
        diagnostics,
        [
            {"example_id": "a", "prediction": "5", "min_top1_prob": 0.8},
            {"example_id": "b", "prediction": "1", "min_top1_prob": 0.1},
        ],
    )
    _write_jsonl(
        target,
        [
            {"example_id": "a", "method": "target_alone", "answer": "5", "prediction": "0", "correct": False},
            {"example_id": "b", "method": "target_alone", "answer": "8", "prediction": "8", "correct": True},
        ],
    )
    target_set.write_text(
        json.dumps(
            {
                "ids": {
                    "clean_residual_targets": ["a"],
                    "target_self_repair": ["b"],
                    "teacher_only": ["a"],
                }
            }
        ),
        encoding="utf-8",
    )

    rows, ids = gate._prepare_rows(
        diagnostics_path=diagnostics,
        target_path=target,
        target_method="target_alone",
        target_set_path=target_set,
        shuffle_offset=1,
        label_shuffle_offset=1,
        outer_folds=2,
    )

    assert [row["example_id"] for row in rows] == ["a", "b"]
    assert rows[0]["source_conditions"]["shuffled_source"]["example_id"] == "b"
    assert ids["clean_residual_targets"] == {"a"}
