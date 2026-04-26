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
