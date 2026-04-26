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


def _write_sidecar(
    path: pathlib.Path,
    *,
    method: str,
    materialized_eval_file: pathlib.Path,
    parsed_args,
) -> None:
    run_config = {
        "source_model": parsed_args.source_model,
        "target_model": parsed_args.target_model,
        "device": parsed_args.device,
        "max_new_tokens": int(parsed_args.max_new_tokens),
        "eval_file": str(materialized_eval_file),
    }
    if method == "c2c":
        run_config["baseline"] = "c2c"
    else:
        thinking_value = "true" if parsed_args.enable_thinking else "false"
        if not parsed_args.use_chat_template:
            thinking_value = "auto"
        run_config.update(
            {
                "translator": str(materializer._resolve(parsed_args.translator)),
                "source_reasoning_mode": parsed_args.source_reasoning_mode,
                "source_use_chat_template": bool(parsed_args.use_chat_template),
                "target_use_chat_template": bool(parsed_args.use_chat_template),
                "source_enable_thinking": thinking_value,
                "target_enable_thinking": thinking_value,
                "methods": [method],
            }
        )
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps({"run_config": run_config}) + "\n",
        encoding="utf-8",
    )


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
    parsed_args = materializer._parse_args(
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
    expected_ids = [
        str(example["example_id"])
        for example in materializer.harness.load_generation(str(materialized))
    ]
    target_output = results_dir / materializer.METHOD_OUTPUTS["target"]
    source_output = results_dir / materializer.METHOD_OUTPUTS["source"]
    _write_jsonl(
        target_output,
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
        source_output,
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
    _write_sidecar(
        target_output,
        method="target",
        materialized_eval_file=materialized,
        parsed_args=parsed_args,
    )
    _write_sidecar(
        source_output,
        method="source",
        materialized_eval_file=materialized,
        parsed_args=parsed_args,
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


def test_validation_rejects_wrong_single_method_artifact(tmp_path: pathlib.Path) -> None:
    eval_file = _toy_eval_file(tmp_path / "eval.jsonl")
    results_dir = tmp_path / "results"
    materialized = results_dir / "_artifacts" / "eval_1.jsonl"
    materializer.harness.materialize_slice(eval_file, materialized, 1)
    parsed_args = materializer._parse_args(
        [
            "--eval-file",
            str(eval_file),
            "--results-dir",
            str(results_dir),
            "--limit",
            "1",
            "--methods",
            "target",
            "--dry-run",
        ]
    )
    expected_ids = [
        str(example["example_id"])
        for example in materializer.harness.load_generation(str(materialized))
    ]
    target_output = results_dir / materializer.METHOD_OUTPUTS["target"]
    _write_jsonl(
        target_output,
        [
            _prediction_record(
                example_id=expected_ids[0],
                method="source_alone",
                correct=True,
                index=0,
            )
        ],
    )
    _write_sidecar(
        target_output,
        method="target",
        materialized_eval_file=materialized,
        parsed_args=parsed_args,
    )

    valid, reason, summary = materializer._validate_record_file(
        method="target",
        path=target_output,
        expected_ids=expected_ids,
        materialized_eval_file=materialized,
        args=parsed_args,
    )

    assert valid is False
    assert reason.startswith("invalid:")
    assert summary is None


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


def test_no_chat_template_sidecar_validation_expects_auto_thinking(tmp_path: pathlib.Path) -> None:
    eval_file = _toy_eval_file(tmp_path / "eval.jsonl")
    results_dir = tmp_path / "results"
    materialized = results_dir / "_artifacts" / "eval_1.jsonl"
    materializer.harness.materialize_slice(eval_file, materialized, 1)
    parsed_args = materializer._parse_args(
        [
            "--eval-file",
            str(eval_file),
            "--results-dir",
            str(results_dir),
            "--limit",
            "1",
            "--methods",
            "source",
            "--no-use-chat-template",
            "--no-enable-thinking",
        ]
    )
    expected_ids = [
        str(example["example_id"])
        for example in materializer.harness.load_generation(str(materialized))
    ]
    source_output = results_dir / materializer.METHOD_OUTPUTS["source"]
    _write_jsonl(
        source_output,
        [
            _prediction_record(
                example_id=expected_ids[0],
                method="source_alone",
                correct=True,
                index=0,
            )
        ],
    )
    _write_sidecar(
        source_output,
        method="source",
        materialized_eval_file=materialized,
        parsed_args=parsed_args,
    )

    valid, reason, summary = materializer._validate_record_file(
        method="source",
        path=source_output,
        expected_ids=expected_ids,
        materialized_eval_file=materialized,
        args=parsed_args,
    )

    assert valid is True
    assert reason == "ok"
    assert summary is not None
    assert summary["sidecar_config_validation"] == "ok"
