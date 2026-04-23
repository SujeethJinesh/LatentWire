from __future__ import annotations

import pathlib
from typing import Any

from scripts import run_gsm8k_contract_residual_sweep as sweep


def test_residual_sweep_uses_shared_runtime_helpers(tmp_path: pathlib.Path, monkeypatch) -> None:
    calibrate_commands: list[list[str]] = []
    evaluate_commands: list[list[str]] = []

    monkeypatch.setattr(sweep.harness, "python_executable", lambda root=sweep.ROOT: "/sentinel/python")
    monkeypatch.setattr(
        sweep.harness,
        "chat_template_cli_args",
        lambda *, enabled, thinking: ["--sentinel-chat", f"enabled={enabled}", f"thinking={thinking}"],
    )
    monkeypatch.setattr(
        sweep,
        "_checkpoint_finite_summary",
        lambda checkpoint_path: {
            "tensor_keys": 1,
            "floating_tensor_keys": 1,
            "total_numel": 1,
            "nonfinite_numel": 0,
            "max_abs": 1.0,
            "first_bad_key": None,
            "nonfinite_keys": [],
            "top_abs_tensors": [],
        },
    )
    monkeypatch.setattr(sweep, "_run", lambda cmd: calibrate_commands.append(cmd))

    config = sweep.ResidualSweepConfig(seed=5)
    checkpoint_path = tmp_path / "checkpoint.pt"
    summary = sweep._calibrate_checkpoint(
        base_label="dynalign_preserve_module_replace",
        rank=16,
        checkpoint_path=checkpoint_path,
        config=config,
    )

    assert summary["nonfinite_numel"] == 0
    assert calibrate_commands == [
        [
            "/sentinel/python",
            str(sweep.ROOT / "scripts" / "calibrate.py"),
            "--calibration-file",
            str(sweep.ROOT / config.calibration_file),
            "--source-model",
            config.source_model,
            "--target-model",
            config.target_model,
            "--output",
            str(checkpoint_path),
            "--bits",
            str(config.bits),
            "--alignment",
            config.alignment,
            "--ridge-lambda",
            str(config.ridge_lambda),
            "--transport-residual-rank",
            str(config.transport_residual_rank),
            "--transport-temperature",
            str(config.transport_temperature),
            "--transport-sinkhorn-iters",
            str(config.transport_sinkhorn_iters),
            "--transport-signature-rank",
            str(config.transport_signature_rank),
            "--transport-signature-weight",
            str(config.transport_signature_weight),
            "--quantization-correction",
            str(sweep.DEFAULT_BASES["dynalign_preserve_module_replace"]["quantization_correction"]),
            "--quantization-correction-rank",
            "16",
            "--bridge-bank-size",
            str(config.bridge_bank_size),
            "--source-reasoning-mode",
            config.source_reasoning_mode,
            "--device",
            config.device,
            "--dtype",
            config.dtype,
            "--seed",
            "5",
            "--sentinel-chat",
            "enabled=True",
            "thinking=False",
        ]
    ]

    monkeypatch.setattr(sweep, "_run", lambda cmd: evaluate_commands.append(cmd))
    monkeypatch.setattr(sweep.smoke, "_read_jsonl", lambda path: [])
    monkeypatch.setattr(
        sweep.smoke,
        "_attach_prompts",
        lambda records, eval_examples_path: [
            {"method": "rotalign_kv", "example_id": "ex1", "correct": 1, "prediction": "42"}
        ],
    )
    monkeypatch.setattr(sweep.smoke, "_group_by_method", lambda records: {"rotalign_kv": records})
    monkeypatch.setattr(
        sweep.checkpoint_sweep,
        "_candidate_row",
        lambda **kwargs: {
            "label": kwargs["label"],
            "accuracy": 1.0,
            "n": 1,
            "correct": 1,
            "example_ids": ["ex1"],
            "numeric_extraction_coverage": 1,
            "empty_predictions": 0,
            "paired_vs_target": {"win": 1, "loss": 0, "tie": 0},
        },
    )

    results_dir = tmp_path / "results"
    row, checks = sweep._run_candidate(
        base_label="dynalign_preserve_module_replace",
        rank=16,
        checkpoint_path=checkpoint_path,
        checkpoint_summary=summary,
        config=config,
        materialized_eval_file=tmp_path / "eval.jsonl",
        baseline_target_records=[{"example_id": "ex1", "correct": 0}],
        results_dir=results_dir,
    )

    prediction_output = results_dir / "dynalign_preserve_module_replace_residrank16.jsonl"
    assert evaluate_commands == [
        [
            "/sentinel/python",
            str(sweep.ROOT / "latent_bridge" / "evaluate.py"),
            "--translator",
            str(checkpoint_path),
            "--source-model",
            config.source_model,
            "--target-model",
            config.target_model,
            "--eval-file",
            str(tmp_path / "eval.jsonl"),
            "--task-type",
            "generation",
            "--device",
            config.device,
            "--max-new-tokens",
            str(config.max_new_tokens),
            "--source-reasoning-mode",
            config.source_reasoning_mode,
            "--kv-transport",
            config.kv_transport,
            "--position-selection-ratio",
            str(config.position_selection_ratio),
            "--position-selection-metric",
            config.position_selection_metric,
            "--gate-mode",
            "fixed",
            "--fixed-gate",
            f"{config.gate:.2f}",
            "--methods",
            "rotalign",
            "--prediction-output",
            str(prediction_output),
            "--random-salt",
            "5",
            "--sentinel-chat",
            "enabled=True",
            "thinking=False",
        ]
    ]
    assert row["label"] == "dynalign_preserve_module_replace_residrank16"
    assert row["seed"] == 5
    assert checks["beats_target"] is True


def test_run_sweep_defaults_materialized_eval_file_to_artifacts_dir(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    captured: dict[str, Any] = {}

    def _fake_materialize(src: pathlib.Path, dst: pathlib.Path, limit: int) -> None:
        captured["materialized_src"] = src
        captured["materialized_dst"] = dst
        captured["materialized_limit"] = limit
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("")

    monkeypatch.setattr(sweep.harness, "materialize_slice", _fake_materialize)
    monkeypatch.setattr(
        sweep.checkpoint_sweep,
        "_load_baseline_target_records",
        lambda results_dir, materialized_eval_file: [{"example_id": "ex1", "correct": 0}],
    )
    monkeypatch.setattr(
        sweep,
        "_calibrate_checkpoint",
        lambda **kwargs: {
            "nonfinite_numel": 0,
            "first_bad_key": None,
            "max_abs": 0.0,
            "top_abs_tensors": [],
        },
    )
    monkeypatch.setattr(
        sweep,
        "_run_candidate",
        lambda **kwargs: (
            {
                "label": "dynalign_module_replace_residrank16",
                "base_label": "dynalign_module_replace",
                "residual_rank": 16,
                "accuracy": 0.125,
                "n": 3,
                "correct": 1,
                "example_ids": ["ex1"],
                "numeric_extraction_coverage": 3,
                "empty_predictions": 0,
                "paired_vs_target": {"win": 1, "loss": 0, "tie": 2},
                "seed": 0,
                "reused_existing_checkpoint": False,
                "status": "ok",
                "checkpoint_nonfinite_numel": 0,
                "checkpoint_first_bad_key": None,
                "checkpoint_max_abs": 0.0,
                "checkpoint_summary": {
                    "nonfinite_numel": 0,
                    "first_bad_key": None,
                    "max_abs": 0.0,
                    "top_abs_tensors": [],
                },
            },
            {
                "row_count_matches_slice": True,
                "example_ids_match_target": True,
                "no_empty_predictions": True,
                "numeric_extraction_coverage": True,
                "beats_target": True,
            },
        ),
    )

    config = sweep.ResidualSweepConfig(
        eval_file="data/gsm8k_eval_70.jsonl",
        slice_size=3,
        materialized_eval_file=None,
        baseline_results_dir=str(tmp_path / "baseline"),
        results_dir=str(tmp_path / "results"),
        checkpoints_dir=str(tmp_path / "checkpoints"),
        bases=("dynalign_module_replace",),
        ranks=(16,),
    )
    payload = sweep.run_sweep(config)

    expected_materialized = tmp_path / "results" / "_artifacts" / "gsm8k_eval_3.jsonl"
    assert captured["materialized_src"] == sweep.ROOT / config.eval_file
    assert captured["materialized_dst"] == expected_materialized
    assert captured["materialized_limit"] == 3
    assert payload["artifacts"]["materialized_eval_file"] == str(expected_materialized)
