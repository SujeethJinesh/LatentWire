from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import json

import latent_bridge.control_suite as control_suite


def test_default_specs_cover_the_control_axes() -> None:
    calibration = control_suite.default_calibration_specs()
    eval_specs = control_suite.default_eval_specs()

    assert [spec.name for spec in calibration] == [
        "interp_full_seed0",
        "cka_half_seed0",
        "cka_quarter_seed0",
        "cka_half_seed1",
    ]
    assert {(spec.layer_pairing, spec.selection_ratio, spec.seed) for spec in calibration} == {
        ("interp", 1.0, 0),
        ("cka", 0.5, 0),
        ("cka", 0.25, 0),
        ("cka", 0.5, 1),
    }
    assert [spec.name for spec in eval_specs] == [
        "fused_noquant_plain",
        "fused_noquant_cot",
        "text_kv_noquant_brief",
        "fused_quant_brief",
    ]
    assert eval_specs[0].include_baselines is True
    assert {spec.source_reasoning_mode for spec in eval_specs} == {
        "plain",
        "cot",
        "brief_analysis",
    }


def test_build_calibrate_cmd_includes_sparse_pairing_and_seed() -> None:
    spec = control_suite.CalibrationSpec("cka_half_seed1", "cka", 0.5, 1)
    cmd = control_suite.build_calibrate_cmd(
        python_exe="python",
        repo_root=Path("/repo"),
        source_model="src",
        target_model="tgt",
        calibration_file="cal.txt",
        checkpoint_path=Path("/tmp/checkpoint.pt"),
        bits=4,
        rotation="orthogonal",
        alignment="ridge",
        whitening=True,
        device="mps",
        dtype="float32",
        spec=spec,
    )

    assert cmd == [
        "python",
        "/repo/scripts/calibrate.py",
        "--source-model",
        "src",
        "--target-model",
        "tgt",
        "--calibration-file",
        "cal.txt",
        "--output",
        "/tmp/checkpoint.pt",
        "--bits",
        "4",
        "--rotation",
        "orthogonal",
        "--alignment",
        "ridge",
        "--layer-pairing",
        "cka",
        "--layer-selection-ratio",
        "0.5",
        "--seed",
        "1",
        "--device",
        "mps",
        "--dtype",
        "float32",
        "--whitening",
    ]


def test_build_evaluate_cmd_includes_baselines_gates_and_reasoning_mode() -> None:
    spec = control_suite.EvalSpec(
        name="fused_noquant_plain",
        methods=("rotalign",),
        gate_values=(0.15, 0.25, 0.30),
        quantize=False,
        source_reasoning_mode="plain",
        include_baselines=True,
    )
    cmd = control_suite.build_evaluate_cmd(
        python_exe="python",
        repo_root=Path("/repo"),
        source_model="src",
        target_model="tgt",
        eval_file="eval.jsonl",
        checkpoint_path=Path("/tmp/checkpoint.pt"),
        task_type="generation",
        device="mps",
        dtype="float32",
        max_new_tokens=64,
        spec=spec,
    )

    assert cmd[:2] == ["python", "/repo/scripts/evaluate.py"]
    assert "--methods" in cmd
    methods_index = cmd.index("--methods")
    assert cmd[methods_index + 1 : methods_index + 4] == ["target", "t2t", "rotalign"]
    assert cmd[cmd.index("--source-reasoning-mode") + 1] == "plain"
    assert cmd[cmd.index("--gate-mode") + 1] == "sweep"
    assert cmd[cmd.index("--gate-values") + 1 : cmd.index("--gate-values") + 4] == [
        "0.15",
        "0.25",
        "0.3",
    ]
    assert "--no-quantize" in cmd


def test_best_metric_for_eval_ignores_system_metrics_and_picks_best_result() -> None:
    metrics = {
        "target_alone": 0.12,
        "text_to_text": 0.09,
        "rotalign_kv_gate_0.15": 0.29,
        "rotalign_kv_gate_0.25": 0.33,
        "rotalign_kv_gate_0.25_bytes": 999.0,
        "rotalign_text_kv_hybrid_gate_0.25": 0.31,
        "rotalign_translated_only_gate_0.25": 0.30,
    }

    best_metric, best_value = control_suite.best_metric_for_eval(
        metrics,
        methods=("rotalign", "rotalign_text_kv", "rotalign_translated"),
        gate_values=(0.15, 0.25, 0.30),
    )

    assert best_metric == "rotalign_kv_gate_0.25"
    assert best_value == 0.33


def test_best_metric_for_eval_can_rank_baselines_with_evaluate_metric_names() -> None:
    metrics = {
        "target_alone": 0.12,
        "text_to_text": 0.41,
        "rotalign_kv_gate_0.25": 0.33,
        "rotalign_kv_gate_0.25_bytes": 999.0,
    }

    best_metric, best_value = control_suite.best_metric_for_eval(
        metrics,
        methods=("t2t", "rotalign"),
        gate_values=(0.25,),
        include_baselines=True,
    )

    assert best_metric == "text_to_text"
    assert best_value == 0.41


def test_control_suite_dry_run_writes_plan_and_skips_subprocesses(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(control_suite.sys, "executable", "python")
    monkeypatch.setattr(
        control_suite,
        "default_calibration_specs",
        lambda: [control_suite.CalibrationSpec("ckpt", "cka", 0.5, 0)],
    )
    monkeypatch.setattr(
        control_suite,
        "default_eval_specs",
        lambda: [
            control_suite.EvalSpec(
                name="eval",
                methods=("rotalign",),
                gate_values=(0.25,),
                quantize=False,
                source_reasoning_mode="cot",
                include_baselines=False,
            )
        ],
    )

    def fail_if_called(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("dry-run should not execute subprocesses")

    monkeypatch.setattr(control_suite, "run_logged_command", fail_if_called)
    monkeypatch.setattr(
        control_suite,
        "parse_args",
        lambda: Namespace(
            source_model="src",
            target_model="tgt",
            calibration_file="cal.txt",
            eval_file="eval.jsonl",
            results_dir=str(tmp_path / "results"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            budget_hours=0.1,
            bits=4,
            rotation="orthogonal",
            alignment="auto",
            whitening=True,
            task_type="generation",
            device="cpu",
            dtype="float32",
            max_new_tokens=64,
            dry_run=True,
        ),
    )

    control_suite.main()

    out = capsys.readouterr().out
    assert "scripts/calibrate.py" in out
    assert "scripts/evaluate.py" in out

    plan_path = tmp_path / "results" / "plan.json"
    assert plan_path.exists()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["source_model"] == "src"
    assert plan["eval_specs"][0]["source_reasoning_mode"] == "cot"
    assert not any((tmp_path / "checkpoints").iterdir())
