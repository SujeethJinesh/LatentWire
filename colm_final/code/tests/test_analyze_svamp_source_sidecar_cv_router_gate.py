import json

from scripts import analyze_svamp32_syndrome_sidecar_probe as syndrome
from scripts import analyze_svamp_source_sidecar_cv_router_gate as cv_gate


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(example_id, method, answer, prediction, *, generated_tokens=64):
    return {
        "example_id": example_id,
        "method": method,
        "answer": str(answer),
        "prediction": str(prediction),
        "normalized_prediction": str(prediction),
        "correct": str(answer) == str(prediction),
        "generated_tokens": generated_tokens,
    }


def test_fold_for_id_is_deterministic_and_bounded():
    values = [cv_gate._fold_for_id("example", 5) for _ in range(3)]
    assert values[0] == values[1] == values[2]
    assert 0 <= values[0] < 5


def test_fit_stump_uses_only_training_folds():
    rows = [
        {
            "fold": 0,
            "features": {"matched": {"source_target_len_ratio": 0.1}},
            "raw_conditions": {"matched": {"correct": True}},
            "fallback_correct": False,
        },
        {
            "fold": 1,
            "features": {"matched": {"source_target_len_ratio": 2.0}},
            "raw_conditions": {"matched": {"correct": False}},
            "fallback_correct": True,
        },
    ]
    rule = cv_gate._fit_stump(
        rows,
        train_folds={0},
        features=["source_target_len_ratio"],
        accept_penalty=0.0,
    )
    assert rule["threshold"] == 0.1


def test_cv_router_gate_emits_predictions_and_controls(tmp_path):
    target_path = tmp_path / "target.jsonl"
    source_path = tmp_path / "source.jsonl"
    target_set_path = tmp_path / "target_set.json"

    ids = ["train_help", "train_harm", "test_help", "test_preserve"]
    answers = {
        "train_help": "5",
        "train_harm": "8",
        "test_help": "7",
        "test_preserve": "9",
    }
    _write_jsonl(
        target_path,
        [
            {**_row("train_help", "target", "5", "0"), "prediction": "wrong target is 0 with a long explanation"},
            {**_row("train_harm", "target", "8", "8"), "prediction": "8"},
            {**_row("test_help", "target", "7", "0"), "prediction": "wrong target is 0 with a long explanation"},
            {**_row("test_preserve", "target", "9", "9"), "prediction": "9"},
        ],
    )
    _write_jsonl(
        source_path,
        [
            {**_row("train_help", "source", "5", "5"), "prediction": "5"},
            {**_row("train_harm", "source", "8", "1"), "prediction": "a very long wrong source answer with 1"},
            {**_row("test_help", "source", "7", "7"), "prediction": "7"},
            {**_row("test_preserve", "source", "9", "1"), "prediction": "a very long wrong source answer with 1"},
        ],
    )
    target_set_path.write_text(
        json.dumps(
            {
                "ids": {
                    "teacher_only": ["train_help", "test_help"],
                    "clean_residual_targets": ["train_help", "test_help"],
                    "target_self_repair": ["train_harm", "test_preserve"],
                }
            }
        ),
        encoding="utf-8",
    )

    payload = cv_gate.analyze(
        target_spec=syndrome.RowSpec("target", target_path, "target"),
        source_spec=syndrome.RowSpec("source", source_path, "source"),
        candidate_specs=[syndrome.RowSpec("source", source_path, "source")],
        target_set_path=target_set_path,
        moduli_sets=[[97]],
        fallback_label="target",
        shuffle_offset=1,
        label_shuffle_offset=1,
        noise_seed=1,
        outer_folds=2,
        features=["source_target_len_ratio"],
        accept_penalty=0.0,
        min_correct=3,
        min_target_self=1,
        min_clean_source_necessary=1,
        max_control_clean_union=2,
        min_numeric_coverage=4,
        run_date="2026-04-26",
    )

    run = payload["runs"][0]
    assert "fold_rules" in run
    assert run["condition_summaries"]["matched"]["correct_count"] >= 3
    assert all(rule["fold"] not in rule["train_folds"] for rule in run["fold_rules"])

    prediction_path = tmp_path / "predictions.jsonl"
    cv_gate._write_predictions(prediction_path, payload, method="cv_router")
    records = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [record["index"] for record in records] == [0, 1, 2, 3]
    assert all("router_rule" in record for record in records)
