import json

from scripts import analyze_svamp70_source_likelihood_sketch_gate as gate


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_sketch_from_scores_quantizes_top_label_and_margin():
    sketch = gate._sketch_from_scores(
        {
            "example_id": "x",
            "candidate_scores": [
                {"label": "target", "score": -2.0},
                {"label": "source", "score": -1.0},
                {"label": "text", "score": -1.5},
            ],
        },
        max_sidecar_bits=8,
    )

    assert sketch["top_label"] == "source"
    assert sketch["margin"] == 0.5
    assert sketch["top_is_source"] == 1.0
    assert 0 <= sketch["quantized_margin"] <= 63
    assert sketch["sidecar_bits"] == 8


def test_fit_stump_accepts_helpful_source_likelihood_sketch():
    rows = [
        {
            "fold": 0,
            "fallback_correct": False,
            "fallback_label": "target",
            "candidate_rows": {
                "target": {"prediction": "0", "correct": False},
                "source": {"prediction": "5", "correct": True},
            },
            "sketch_conditions": {
                "matched": {"top_label": "source", "margin": 3.0, "top_is_source": 1.0, "sidecar_bits": 8},
            },
        },
        {
            "fold": 1,
            "fallback_correct": True,
            "fallback_label": "target",
            "candidate_rows": {
                "target": {"prediction": "8", "correct": True},
                "source": {"prediction": "1", "correct": False},
            },
            "sketch_conditions": {
                "matched": {"top_label": "source", "margin": -1.0, "top_is_source": 1.0, "sidecar_bits": 8},
            },
        },
    ]

    rule = gate._fit_stump(rows, train_folds=None, accept_penalty=0.0)

    assert rule["feature"] == "margin"
    assert rule["direction"] == "ge"
    assert -1.0 < rule["threshold"] <= 3.0
    assert rule["train_help"] == 1
    assert rule["train_harm"] == 0


def test_evaluate_counts_source_necessary_and_controls():
    rows = [
        {
            "index": 0,
            "example_id": "clean",
            "fold": 0,
            "fallback_label": "target",
            "fallback_correct": False,
            "candidate_rows": {
                "target": {"prediction": "0", "correct": False},
                "source": {"prediction": "5", "correct": True},
            },
            "sketch_conditions": {
                "matched": {"top_label": "source", "margin": 2.0, "sidecar_bits": 8},
                "zero_source": None,
                "shuffled_source": {"top_label": "target", "margin": 2.0, "sidecar_bits": 8},
                "label_shuffle": {"top_label": "target", "margin": 2.0, "sidecar_bits": 8},
                "target_only": None,
                "slots_only": None,
            },
        },
        {
            "index": 1,
            "example_id": "preserve",
            "fold": 0,
            "fallback_label": "target",
            "fallback_correct": True,
            "candidate_rows": {
                "target": {"prediction": "8", "correct": True},
                "source": {"prediction": "1", "correct": False},
            },
            "sketch_conditions": {
                "matched": {"top_label": "source", "margin": -1.0, "sidecar_bits": 8},
                "zero_source": None,
                "shuffled_source": {"top_label": "source", "margin": 2.0, "sidecar_bits": 8},
                "label_shuffle": {"top_label": "source", "margin": 2.0, "sidecar_bits": 8},
                "target_only": None,
                "slots_only": None,
            },
        },
    ]
    ids = {
        "clean_residual_targets": {"clean"},
        "target_self_repair": {"preserve"},
        "clean_source_only": {"clean"},
    }
    rule = {"feature": "margin", "threshold": 0.5, "direction": "ge"}

    result = gate._evaluate(rows=rows, target_ids=ids, global_rule=rule)

    assert result["condition_summaries"]["matched"]["correct_count"] == 2
    assert result["source_necessary_clean_ids"] == ["clean"]
    assert result["control_clean_union_ids"] == []
    assert result["accepted_harm"] == 0


def test_prepare_rows_builds_shuffled_sketch_controls(tmp_path):
    sketch = tmp_path / "sketch.jsonl"
    target = tmp_path / "target.jsonl"
    source = tmp_path / "source.jsonl"
    target_set = tmp_path / "target_set.json"
    _write_jsonl(
        sketch,
        [
            {"example_id": "a", "candidate_scores": [{"label": "source", "score": 2.0}, {"label": "target", "score": 0.0}]},
            {"example_id": "b", "candidate_scores": [{"label": "target", "score": 2.0}, {"label": "source", "score": 0.0}]},
        ],
    )
    _write_jsonl(
        target,
        [
            {"example_id": "a", "method": "target_alone", "answer": "5", "prediction": "0", "correct": False},
            {"example_id": "b", "method": "target_alone", "answer": "8", "prediction": "8", "correct": True},
        ],
    )
    _write_jsonl(
        source,
        [
            {"example_id": "a", "method": "source_alone", "answer": "5", "prediction": "5", "correct": True},
            {"example_id": "b", "method": "source_alone", "answer": "8", "prediction": "1", "correct": False},
        ],
    )
    target_set.write_text(
        json.dumps(
            {
                "ids": {
                    "clean_residual_targets": ["a"],
                    "clean_source_only": ["a"],
                    "target_self_repair": ["b"],
                }
            }
        ),
        encoding="utf-8",
    )

    rows, ids = gate._prepare_rows(
        sketch_path=sketch,
        target_set_path=target_set,
        candidate_specs=[
            gate.syndrome.RowSpec("target", target, "target_alone"),
            gate.syndrome.RowSpec("source", source, "source_alone"),
        ],
        fallback_label="target",
        shuffle_offset=1,
        label_shuffle_offset=1,
        outer_folds=2,
        max_sidecar_bits=8,
    )

    assert [row["example_id"] for row in rows] == ["a", "b"]
    assert rows[0]["sketch_conditions"]["matched"]["top_label"] == "source"
    assert rows[0]["sketch_conditions"]["shuffled_source"]["top_label"] == "target"
    assert ids["clean_residual_targets"] == {"a"}
