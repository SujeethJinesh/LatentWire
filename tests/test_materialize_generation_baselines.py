from __future__ import annotations

import json
import pathlib

from scripts import materialize_generation_baselines as materializer


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def _toy_eval_file(path: pathlib.Path) -> pathlib.Path:
    _write_jsonl(
        path,
        [
            {"question": "q0", "answer": "1", "aliases": ["#### 1"]},
            {"question": "q1", "answer": "2", "aliases": ["#### 2"]},
        ],
    )
    return path


def _prediction_record(
    *,
    example_id: str,
    method: str,
    correct: bool,
    index: int,
) -> dict:
    return {
        "answer": ["1"],
        "correct": correct,
        "example_id": example_id,
        "index": index,
        "method": method,
        "normalized_prediction": "1" if correct else "0",
        "prediction": "answer is 1" if correct else "answer is 0",
    }


def test_dry_run_writes_materialized_slice_and_commands(tmp_path: pathlib.Path) -> None:
    eval_file = _toy_eval_file(tmp_path / "eval.jsonl")
    results_dir = tmp_path / "results"

    payload = materializer.main(
        [
            "--eval-file",
            str(eval_file),
            "--results-dir",
            str(results_dir),
            "--limit",
            "1",
            "--methods",
            "source",
            "c2c",
            "--dry-run",
        ]
    )

    materialized = results_dir / "_artifacts" / "eval_1.jsonl"
    assert materialized.exists()
    assert materialized.read_text(encoding="utf-8").count("\n") == 1
    assert [row["status"] for row in payload["runs"]] == ["dry_run", "dry_run"]
    assert "latent_bridge/evaluate.py" in " ".join(payload["runs"][0]["command"])
    assert "scripts/run_c2c_eval.py" in " ".join(payload["runs"][1]["command"])
    assert (results_dir / "manifest.json").exists()
    assert (results_dir / "manifest.md").exists()


def test_existing_outputs_are_summarized_with_exact_id_pairing(tmp_path: pathlib.Path) -> None:
    eval_file = _toy_eval_file(tmp_path / "eval.jsonl")
    results_dir = tmp_path / "results"
    materialized = results_dir / "_artifacts" / "eval_2.jsonl"
    materializer.harness.materialize_slice(eval_file, materialized, 2)
    expected_ids = [
        str(example["example_id"])
        for example in materializer.harness.load_generation(str(materialized))
    ]
    _write_jsonl(
        results_dir / materializer.METHOD_OUTPUTS["target"],
        [
            _prediction_record(
                example_id=expected_ids[0],
                method="target_alone",
                correct=True,
                index=0,
            ),
            _prediction_record(
                example_id=expected_ids[1],
                method="target_alone",
                correct=False,
                index=1,
            ),
        ],
    )
    _write_jsonl(
        results_dir / materializer.METHOD_OUTPUTS["source"],
        [
            _prediction_record(
                example_id=expected_ids[0],
                method="source_alone",
                correct=False,
                index=0,
            ),
            _prediction_record(
                example_id=expected_ids[1],
                method="source_alone",
                correct=True,
                index=1,
            ),
        ],
    )

    payload = materializer.main(
        [
            "--eval-file",
            str(eval_file),
            "--results-dir",
            str(results_dir),
            "--limit",
            "2",
            "--methods",
            "target",
            "source",
            "--dry-run",
        ]
    )

    assert [row["status"] for row in payload["runs"]] == [
        "skipped_existing",
        "skipped_existing",
    ]
    assert payload["method_summaries"]["target"]["exact_id_parity"] is True
    assert payload["method_summaries"]["source"]["exact_id_parity"] is True
    assert payload["pairwise_vs_target"][0]["method"] == "source"
    assert payload["pairwise_vs_target"][0]["method_only_count"] == 1
    assert payload["pairwise_vs_target"][0]["oracle_count"] == 2


def test_generation_eval_command_uses_single_method_output(tmp_path: pathlib.Path) -> None:
    args = materializer._parse_args(
        [
            "--eval-file",
            str(tmp_path / "eval.jsonl"),
            "--results-dir",
            str(tmp_path / "results"),
            "--methods",
            "target",
            "--limit",
            "1",
        ]
    )
    command = materializer.build_generation_eval_command(
        method="target",
        eval_file=tmp_path / "eval_1.jsonl",
        prediction_output=tmp_path / "target.jsonl",
        args=args,
    )

    method_index = command.index("--methods")
    output_index = command.index("--prediction-output")
    assert command[method_index + 1] == "target"
    assert pathlib.Path(command[output_index + 1]).name == "target.jsonl"
    assert "--source-use-chat-template" in command
    assert "--target-enable-thinking" in command
