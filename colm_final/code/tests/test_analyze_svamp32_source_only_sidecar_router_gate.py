import json

from scripts import analyze_svamp32_source_only_sidecar_router_gate as gate
from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(example_id, method, answer, prediction):
    return {
        "example_id": example_id,
        "method": method,
        "answer": str(answer),
        "prediction": str(prediction),
        "normalized_prediction": str(prediction),
        "correct": str(answer) == str(prediction),
    }


def test_source_only_sidecar_router_can_clear_synthetic_gate(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    target_self_path = tmp_path / "target_self.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    target_set_path = tmp_path / "target_set.json"

    ids = ["clean", "preserve", "other"]
    answers = {"clean": "5", "preserve": "8", "other": "11"}
    _write_jsonl(
        target_path,
        [_row(example_id, "target_alone", answers[example_id], "0") for example_id in ids],
    )
    _write_jsonl(
        source_path,
        [
            _row("clean", "source_alone", "5", "5"),
            _row("preserve", "source_alone", "8", "8"),
            _row("other", "source_alone", "11", "4"),
        ],
    )
    _write_jsonl(
        target_self_path,
        [
            _row("clean", "target_self_repair", "5", "0"),
            _row("preserve", "target_self_repair", "8", "8"),
            _row("other", "target_self_repair", "11", "0"),
        ],
    )
    _write_jsonl(
        candidate_path,
        [
            _row("clean", "candidate", "5", "5"),
            _row("preserve", "candidate", "8", "8"),
            _row("other", "candidate", "11", "0"),
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["clean"],
                    "clean_residual_targets": ["clean"],
                    "target_self_repair": ["preserve"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = gate.analyze(
        target_spec=syndrome.RowSpec("target_alone", target_path, "target_alone"),
        source_spec=syndrome.RowSpec("source_alone", source_path, "source_alone"),
        candidate_specs=[
            syndrome.RowSpec("target_self_repair", target_self_path, "target_self_repair"),
            syndrome.RowSpec("candidate", candidate_path, "candidate"),
        ],
        target_set_path=target_set_path,
        moduli_sets=[[7]],
        fallback_label="target_self_repair",
        shuffle_offset=1,
        label_shuffle_offset=2,
        noise_seed=1,
        min_correct=2,
        min_target_self=1,
        min_clean_source_necessary=1,
        max_control_clean_union=0,
        min_numeric_coverage=3,
        run_date="2026-04-26",
    )

    run = payload["runs"][0]
    assert payload["status"] == "source_only_sidecar_router_clears_gate"
    assert run["condition_summaries"]["matched"]["correct_count"] == 2
    assert run["source_necessary_clean_ids"] == ["clean"]


def test_preserve_on_agreement_guard_blocks_source_harm(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    text_path = tmp_path / "text.jsonl"
    target_set_path = tmp_path / "target_set.json"

    ids = ["clean", "preserve"]
    _write_jsonl(
        target_path,
        [
            _row("clean", "target", "5", "0"),
            _row("preserve", "target", "8", "8"),
        ],
    )
    _write_jsonl(
        source_path,
        [
            _row("clean", "source", "5", "5"),
            _row("preserve", "source", "8", "1"),
        ],
    )
    _write_jsonl(
        text_path,
        [
            _row("clean", "text", "5", "2"),
            _row("preserve", "text", "8", "8"),
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["clean"],
                    "clean_residual_targets": ["clean"],
                    "target_self_repair": ["preserve"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = gate.analyze(
        target_spec=syndrome.RowSpec("target", target_path, "target"),
        source_spec=syndrome.RowSpec("source", source_path, "source"),
        candidate_specs=[
            syndrome.RowSpec("source", source_path, "source"),
            syndrome.RowSpec("text", text_path, "text"),
        ],
        target_set_path=target_set_path,
        moduli_sets=[[7]],
        fallback_label="target",
        shuffle_offset=1,
        label_shuffle_offset=1,
        noise_seed=1,
        min_correct=2,
        min_target_self=1,
        min_clean_source_necessary=1,
        max_control_clean_union=0,
        min_numeric_coverage=2,
        preserve_on_agreement_label="text",
        run_date="2026-04-26",
    )

    run = payload["runs"][0]
    assert payload["status"] == "source_only_sidecar_router_clears_gate"
    assert run["condition_summaries"]["matched"]["correct_count"] == 2
    assert run["source_necessary_clean_ids"] == ["clean"]
    preserve_row = [row for row in run["rows"] if row["example_id"] == "preserve"][0]
    assert preserve_row["agreement_guard_active"] is True
    assert preserve_row["conditions"]["matched"]["prediction"] == "8"


def test_source_quality_guard_blocks_low_quality_source_switch(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    target_set_path = tmp_path / "target_set.json"

    _write_jsonl(
        target_path,
        [
            _row("clean", "target", "5", "0"),
            _row("preserve", "target", "8", "8"),
        ],
    )
    _write_jsonl(
        source_path,
        [
            {
                **_row("clean", "source", "5", "5"),
                "prediction": "Step-by-step explanation: 2 + 3 = 5. Answer: 5",
            },
            {
                **_row("preserve", "source", "8", "1"),
                "prediction": "8 and 1 and 2 and 3 and 4 and 5 and 6 and 7",
            },
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["clean"],
                    "clean_residual_targets": ["clean"],
                    "target_self_repair": ["preserve"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = gate.analyze(
        target_spec=syndrome.RowSpec("target", target_path, "target"),
        source_spec=syndrome.RowSpec("source", source_path, "source"),
        candidate_specs=[syndrome.RowSpec("source", source_path, "source")],
        target_set_path=target_set_path,
        moduli_sets=[[7]],
        fallback_label="target",
        shuffle_offset=1,
        label_shuffle_offset=1,
        noise_seed=1,
        min_correct=2,
        min_target_self=1,
        min_clean_source_necessary=0,
        max_control_clean_union=1,
        min_numeric_coverage=2,
        source_quality_guard="finalish_short_numeric",
        run_date="2026-04-26",
    )

    run = payload["runs"][0]
    assert payload["status"] == "source_only_sidecar_router_clears_gate"
    assert run["condition_summaries"]["matched"]["correct_count"] == 2
    preserve_row = [row for row in run["rows"] if row["example_id"] == "preserve"][0]
    assert preserve_row["source_quality_passed"] is False
    assert preserve_row["conditions"]["matched"]["prediction"] == "8"


def test_shorter_than_target_numeric_guard_uses_source_and_target_lengths(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    target_set_path = tmp_path / "target_set.json"

    _write_jsonl(
        target_path,
        [
            {
                **_row("clean", "target", "5", "0"),
                "prediction": "A long wrong target explanation ending in 0",
            },
            {
                **_row("preserve", "target", "8", "8"),
                "prediction": "8",
            },
        ],
    )
    _write_jsonl(
        source_path,
        [
            {
                **_row("clean", "source", "5", "5"),
                "prediction": "5",
            },
            {
                **_row("preserve", "source", "8", "1"),
                "prediction": "A longer wrong source answer with 1",
            },
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["clean"],
                    "clean_residual_targets": ["clean"],
                    "target_self_repair": ["preserve"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = gate.analyze(
        target_spec=syndrome.RowSpec("target", target_path, "target"),
        source_spec=syndrome.RowSpec("source", source_path, "source"),
        candidate_specs=[syndrome.RowSpec("source", source_path, "source")],
        target_set_path=target_set_path,
        moduli_sets=[[7]],
        fallback_label="target",
        shuffle_offset=1,
        label_shuffle_offset=1,
        noise_seed=1,
        min_correct=2,
        min_target_self=1,
        min_clean_source_necessary=1,
        max_control_clean_union=1,
        min_numeric_coverage=2,
        source_quality_guard="shorter_than_target_numeric",
        run_date="2026-04-26",
    )

    run = payload["runs"][0]
    assert payload["status"] == "source_only_sidecar_router_clears_gate"
    assert run["condition_summaries"]["matched"]["correct_count"] == 2
    clean_row = [row for row in run["rows"] if row["example_id"] == "clean"][0]
    preserve_row = [row for row in run["rows"] if row["example_id"] == "preserve"][0]
    assert clean_row["source_quality_passed"] is True
    assert preserve_row["source_quality_passed"] is False
    assert preserve_row["conditions"]["matched"]["prediction"] == "8"


def test_source_quality_len_ratio_threshold_is_parameterized(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    target_set_path = tmp_path / "target_set.json"

    _write_jsonl(
        target_path,
        [
            {
                **_row("clean", "target", "5", "0"),
                "prediction": "A long wrong target explanation ending in 0",
            },
            {
                **_row("preserve", "target", "8", "8"),
                "prediction": "8",
            },
        ],
    )
    _write_jsonl(
        source_path,
        [
            {
                **_row("clean", "source", "5", "5"),
                "prediction": "5",
            },
            {
                **_row("preserve", "source", "8", "1"),
                "prediction": "A longer wrong source answer with 1",
            },
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["clean"],
                    "clean_residual_targets": ["clean"],
                    "target_self_repair": ["preserve"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = gate.analyze(
        target_spec=syndrome.RowSpec("target", target_path, "target"),
        source_spec=syndrome.RowSpec("source", source_path, "source"),
        candidate_specs=[syndrome.RowSpec("source", source_path, "source")],
        target_set_path=target_set_path,
        moduli_sets=[[7]],
        fallback_label="target",
        shuffle_offset=1,
        label_shuffle_offset=1,
        noise_seed=1,
        min_correct=2,
        min_target_self=1,
        min_clean_source_necessary=1,
        max_control_clean_union=1,
        min_numeric_coverage=2,
        source_quality_score_field="source_target_len_ratio",
        source_quality_max_threshold=1.0,
        run_date="2026-04-26",
    )

    run = payload["runs"][0]
    assert payload["status"] == "source_only_sidecar_router_clears_gate"
    clean_row = [row for row in run["rows"] if row["example_id"] == "clean"][0]
    preserve_row = [row for row in run["rows"] if row["example_id"] == "preserve"][0]
    assert clean_row["source_quality_passed"] is True
    assert clean_row["source_quality_score"] < 1.0
    assert preserve_row["source_quality_passed"] is False
    assert preserve_row["source_quality_score"] > 1.0
    assert preserve_row["conditions"]["matched"]["prediction"] == "8"

    prediction_path = tmp_path / "predictions.jsonl"
    gate._write_prediction_jsonl(
        prediction_path,
        payload,
        method="source_lenratio_sidecar",
    )
    records = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["example_id"] for record in records] == ["clean", "preserve"]
    assert [record["index"] for record in records] == [0, 1]
    assert records[0]["method"] == "source_lenratio_sidecar"
    assert records[0]["correct"] is True
    assert records[0]["sidecar_moduli"] == [7]
    assert records[1]["accepted_source_sidecar"] is False


def test_hash_shuffle_controls_use_nonself_source_ids(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    target_set_path = tmp_path / "target_set.json"
    ids = ["a", "b", "c"]

    _write_jsonl(
        target_path,
        [_row(example_id, "target", "9", "0") for example_id in ids],
    )
    _write_jsonl(
        source_path,
        [
            _row("a", "source", "9", "1"),
            _row("b", "source", "9", "2"),
            _row("c", "source", "9", "3"),
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["a"],
                    "clean_residual_targets": ["a"],
                    "target_self_repair": [],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = gate.analyze(
        target_spec=syndrome.RowSpec("target", target_path, "target"),
        source_spec=syndrome.RowSpec("source", source_path, "source"),
        candidate_specs=[syndrome.RowSpec("source", source_path, "source")],
        target_set_path=target_set_path,
        moduli_sets=[[7]],
        fallback_label="target",
        shuffle_offset=0,
        label_shuffle_offset=0,
        noise_seed=1,
        min_correct=0,
        min_target_self=0,
        min_clean_source_necessary=0,
        max_control_clean_union=3,
        min_numeric_coverage=3,
        shuffle_mode="hash",
        run_date="2026-04-27",
    )

    rows = payload["runs"][0]["rows"]
    for row in rows:
        assert row["shuffled_source_id"] != row["example_id"]
        assert row["label_shuffled_source_id"] != row["example_id"]
    assert payload["config"]["shuffle_mode"] == "hash"
