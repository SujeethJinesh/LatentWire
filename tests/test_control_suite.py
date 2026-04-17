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
        "no_rotation_cka_half_seed1",
        "hadamard_cka_half_seed1",
        "dct_cka_half_seed1",
        "identity_align_cka_half_seed1",
        "shifted_layers_half_seed1",
        "random_layers_half_seed1",
        "lowrank128_cka_half_seed1",
        "cka_half_headhalf_lowrank_affine_seed1",
        "bits2_cka_half_seed1",
        "bits3_cka_half_seed1",
        "bits6_cka_half_seed1",
        "bits8_cka_half_seed1",
    ]
    assert ("cka", 0.5, 1) in {
        (spec.layer_pairing, spec.selection_ratio, spec.seed) for spec in calibration
    }
    assert any(spec.rotation == "dct" for spec in calibration)
    assert any(spec.alignment == "identity" for spec in calibration)
    assert any(spec.layer_pairing == "random" for spec in calibration)
    assert any(spec.bits == 2 for spec in calibration)
    assert [spec.name for spec in eval_specs] == [
        "baseline_plain",
        "baseline_brief",
        "baseline_cot",
        "fused_noquant_plain",
        "fused_noquant_brief",
        "fused_noquant_cot",
        "translated_noquant_brief",
        "text_kv_noquant_brief",
        "fused_quant_brief",
        "fused_quant_cosine_shifted_brief",
        "fused_quant_kalman_brief",
        "fused_quant_noise_brief",
        "fused_quant_random_kv_brief",
        "fused_quant_shuffle_kv_brief",
        "fused_quant_zero_kv_brief",
        "target_attenuation_brief",
        "fused_quant_random_translated_brief",
        "fused_quant_k_only_brief",
        "fused_quant_k_only_cosine_shifted_brief",
        "fused_quant_k_only_cosine_brief",
        "translated_quant_k_only_brief",
        "target_attenuation_k_only_brief",
        "fused_quant_v_only_cosine_brief",
        "translated_quant_v_only_brief",
        "target_attenuation_v_only_brief",
        "fused_quant_k_only_attention_sparse_brief",
    ]
    assert all(spec.include_baselines for spec in eval_specs)
    assert {spec.source_reasoning_mode for spec in eval_specs} == {
        "plain",
        "cot",
        "brief_analysis",
    }


def test_filter_named_specs_preserves_requested_order() -> None:
    specs = control_suite.default_eval_specs()
    filtered = control_suite._filter_named_specs(
        specs,
        ["fused_quant_brief", "baseline_plain", "translated_noquant_brief"],
    )

    assert [spec.name for spec in filtered] == [
        "fused_quant_brief",
        "baseline_plain",
        "translated_noquant_brief",
    ]


def test_filter_named_specs_supports_target_attenuation_alias() -> None:
    filtered = control_suite._filter_named_specs(
        control_suite.default_eval_specs(),
        ["fused_quant_zero_translated_brief"],
    )

    assert [spec.name for spec in filtered] == ["target_attenuation_brief"]


def test_filter_named_specs_rejects_unknown_name() -> None:
    try:
        control_suite._filter_named_specs(control_suite.default_eval_specs(), ["missing"])
    except ValueError as exc:
        assert "Unknown spec name(s): missing" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for unknown spec name")


def test_build_calibrate_cmd_includes_sparse_pairing_and_seed() -> None:
    spec = control_suite.CalibrationSpec(
        "dct_lowrank", "random", 0.5, 1, rotation="dct", alignment="reduced_rank", bits=3, alignment_rank=2
    )
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
        "3",
        "--rotation",
        "dct",
        "--alignment",
        "reduced_rank",
        "--layer-pairing",
        "random",
        "--layer-selection-ratio",
        "0.5",
        "--seed",
        "1",
        "--device",
        "mps",
        "--dtype",
        "float32",
        "--alignment-rank",
        "2",
        "--whitening",
    ]


def test_build_calibrate_cmd_passes_head_prequant_and_correction_flags() -> None:
    spec = control_suite.CalibrationSpec(
        "head_lowrank_affine",
        "cka",
        0.5,
        1,
        head_selection_topk=1,
        head_selection_ratio=0.5,
        head_selection_metric="negative_error",
        pre_quant_rank=64,
        pre_quant_shrinkage=0.25,
        quantization_correction="affine",
    )
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
        whitening=False,
        device="mps",
        dtype="float32",
        spec=spec,
    )

    assert cmd[cmd.index("--head-selection-topk") + 1] == "1"
    assert cmd[cmd.index("--head-selection-ratio") + 1] == "0.5"
    assert cmd[cmd.index("--head-selection-metric") + 1] == "negative_error"
    assert cmd[cmd.index("--pre-quant-rank") + 1] == "64"
    assert cmd[cmd.index("--pre-quant-shrinkage") + 1] == "0.25"
    assert cmd[cmd.index("--quantization-correction") + 1] == "affine"


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
        gate_search_file=None,
        gate_search_limit=30,
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


def test_build_evaluate_cmd_uses_held_out_gate_search_when_requested() -> None:
    spec = control_suite.EvalSpec(
        name="fused_quant_brief",
        methods=("rotalign",),
        gate_values=(0.15, 0.25, 0.30),
        quantize=True,
        source_reasoning_mode="brief_analysis",
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
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=spec,
    )

    assert cmd[cmd.index("--gate-mode") + 1] == "search"
    assert cmd[cmd.index("--gate-search-file") + 1] == "gate.jsonl"
    assert cmd[cmd.index("--gate-search-limit") + 1] == "12"
    assert cmd[cmd.index("--gate-values") + 1 : cmd.index("--gate-values") + 4] == [
        "0.15",
        "0.25",
        "0.3",
    ]


def test_build_evaluate_cmd_passes_nondefault_fusion_rule() -> None:
    spec = control_suite.EvalSpec(
        name="fused_quant_cosine",
        methods=("rotalign",),
        gate_values=(0.15,),
        quantize=True,
        source_reasoning_mode="brief_analysis",
        fusion_rule="cosine",
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
        gate_search_file=None,
        gate_search_limit=30,
        spec=spec,
    )

    assert cmd[cmd.index("--fusion-rule") + 1] == "cosine"


def test_build_evaluate_cmd_passes_nondefault_kv_transport() -> None:
    spec = control_suite.EvalSpec(
        name="fused_quant_k_only",
        methods=("rotalign",),
        gate_values=(0.15,),
        quantize=True,
        source_reasoning_mode="brief_analysis",
        kv_transport="k_only",
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
        gate_search_file=None,
        gate_search_limit=30,
        spec=spec,
    )

    assert cmd[cmd.index("--kv-transport") + 1] == "k_only"


def test_build_evaluate_cmd_passes_nondefault_position_selection_ratio() -> None:
    spec = control_suite.EvalSpec(
        name="fused_quant_sparse_k_only",
        methods=("rotalign",),
        gate_values=(0.10,),
        quantize=True,
        source_reasoning_mode="brief_analysis",
        kv_transport="k_only",
        position_selection_ratio=0.5,
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
        gate_search_file=None,
        gate_search_limit=30,
        spec=spec,
    )

    assert cmd[cmd.index("--position-selection-ratio") + 1] == "0.5"


def test_build_evaluate_cmd_passes_nondefault_position_selection_metric() -> None:
    spec = control_suite.EvalSpec(
        name="fused_quant_attention_sparse_k_only",
        methods=("rotalign",),
        gate_values=(0.10,),
        quantize=True,
        source_reasoning_mode="brief_analysis",
        kv_transport="k_only",
        position_selection_ratio=0.5,
        position_selection_metric="attention",
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
        gate_search_file=None,
        gate_search_limit=30,
        spec=spec,
    )

    assert cmd[cmd.index("--position-selection-metric") + 1] == "attention"


def test_build_evaluate_cmd_skips_noop_gate_search_for_translated_only() -> None:
    spec = control_suite.EvalSpec(
        name="translated_noquant_brief",
        methods=("rotalign_translated",),
        gate_values=(0.15, 0.25, 0.30),
        quantize=False,
        source_reasoning_mode="brief_analysis",
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
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=spec,
    )

    assert cmd[cmd.index("--gate-mode") + 1] == "checkpoint"
    assert "--gate-search-file" not in cmd
    assert "--gate-values" not in cmd
    methods_index = cmd.index("--methods")
    assert cmd[methods_index + 1 : methods_index + 4] == [
        "target",
        "t2t",
        "rotalign_translated",
    ]


def test_build_evaluate_cmd_adds_source_kv_control_and_prediction_output() -> None:
    spec = control_suite.EvalSpec(
        name="fused_quant_random_kv_brief",
        methods=("rotalign",),
        gate_values=(0.15,),
        quantize=True,
        source_reasoning_mode="brief_analysis",
        include_baselines=True,
        source_kv_control="random",
    )
    cmd = control_suite.build_evaluate_cmd(
        python_exe="python",
        repo_root=Path("/repo"),
        source_model="src",
        target_model="tgt",
        eval_file="eval.jsonl",
        checkpoint_path=Path("/tmp/checkpoint.pt"),
        task_type="mcq",
        device="mps",
        dtype="float32",
        max_new_tokens=64,
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=spec,
        prediction_output=Path("/tmp/preds.jsonl"),
    )

    assert cmd[cmd.index("--source-kv-control") + 1] == "random"
    assert cmd[cmd.index("--prediction-output") + 1] == "/tmp/preds.jsonl"


def test_build_evaluate_cmd_adds_quantization_control() -> None:
    spec = control_suite.EvalSpec(
        name="fused_quant_noise_brief",
        methods=("rotalign",),
        gate_values=(0.15,),
        quantize=True,
        source_reasoning_mode="brief_analysis",
        include_baselines=False,
        quantization_control="matched_noise",
    )
    cmd = control_suite.build_evaluate_cmd(
        python_exe="python",
        repo_root=Path("/repo"),
        source_model="src",
        target_model="tgt",
        eval_file="eval.jsonl",
        checkpoint_path=Path("/tmp/checkpoint.pt"),
        task_type="mcq",
        device="mps",
        dtype="float32",
        max_new_tokens=64,
        gate_search_file=None,
        gate_search_limit=12,
        spec=spec,
    )

    assert cmd[cmd.index("--quantization-control") + 1] == "matched_noise"


def test_build_evaluate_cmd_adds_translated_kv_control() -> None:
    spec = control_suite.EvalSpec(
        name="target_attenuation_brief",
        methods=("rotalign",),
        gate_values=(0.15,),
        quantize=True,
        source_reasoning_mode="brief_analysis",
        translated_kv_control="zero",
    )
    cmd = control_suite.build_evaluate_cmd(
        python_exe="python",
        repo_root=Path("/repo"),
        source_model="src",
        target_model="tgt",
        eval_file="eval.jsonl",
        checkpoint_path=Path("/tmp/checkpoint.pt"),
        task_type="mcq",
        device="mps",
        dtype="float32",
        max_new_tokens=64,
        gate_search_file=None,
        gate_search_limit=12,
        spec=spec,
    )

    assert cmd[cmd.index("--translated-kv-control") + 1] == "zero"


def test_target_attenuation_spec_is_zero_byte_control() -> None:
    spec = next(
        spec
        for spec in control_suite.default_eval_specs()
        if spec.name == "target_attenuation_brief"
    )

    assert spec.translated_kv_control == "zero"
    assert spec.methods == ("rotalign",)
    assert spec.gate_values == (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)


def test_k_only_specs_cover_fused_translated_and_zero_byte_controls() -> None:
    specs = {spec.name: spec for spec in control_suite.default_eval_specs()}

    assert specs["fused_quant_k_only_brief"].kv_transport == "k_only"
    assert specs["fused_quant_k_only_cosine_shifted_brief"].fusion_rule == "cosine_shifted"
    assert specs["fused_quant_k_only_cosine_brief"].fusion_rule == "cosine"
    assert specs["translated_quant_k_only_brief"].methods == ("rotalign_translated",)
    assert specs["translated_quant_k_only_brief"].gate_values == ()
    assert specs["target_attenuation_k_only_brief"].translated_kv_control == "zero"
    assert specs["fused_quant_v_only_cosine_brief"].kv_transport == "v_only"
    assert specs["fused_quant_v_only_cosine_brief"].fusion_rule == "cosine"
    assert specs["translated_quant_v_only_brief"].methods == ("rotalign_translated",)
    assert specs["target_attenuation_v_only_brief"].translated_kv_control == "zero"
    assert specs["fused_quant_k_only_attention_sparse_brief"].position_selection_metric == "attention"
    assert specs["fused_quant_k_only_attention_sparse_brief"].position_selection_ratio == 0.5


def test_build_evaluate_cmd_for_k_only_followup_specs() -> None:
    specs = {spec.name: spec for spec in control_suite.default_eval_specs()}

    fused_cmd = control_suite.build_evaluate_cmd(
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
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=specs["fused_quant_k_only_cosine_shifted_brief"],
    )
    assert fused_cmd[fused_cmd.index("--fusion-rule") + 1] == "cosine_shifted"
    assert fused_cmd[fused_cmd.index("--kv-transport") + 1] == "k_only"
    assert fused_cmd[fused_cmd.index("--gate-mode") + 1] == "search"

    translated_cmd = control_suite.build_evaluate_cmd(
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
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=specs["translated_quant_k_only_brief"],
    )
    assert translated_cmd[translated_cmd.index("--kv-transport") + 1] == "k_only"
    assert translated_cmd[translated_cmd.index("--gate-mode") + 1] == "checkpoint"
    assert "--gate-values" not in translated_cmd

    attenuation_cmd = control_suite.build_evaluate_cmd(
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
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=specs["target_attenuation_k_only_brief"],
    )
    assert attenuation_cmd[attenuation_cmd.index("--translated-kv-control") + 1] == "zero"
    assert attenuation_cmd[attenuation_cmd.index("--kv-transport") + 1] == "k_only"

    cosine_cmd = control_suite.build_evaluate_cmd(
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
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=specs["fused_quant_k_only_cosine_brief"],
    )
    assert cosine_cmd[cosine_cmd.index("--fusion-rule") + 1] == "cosine"
    assert cosine_cmd[cosine_cmd.index("--kv-transport") + 1] == "k_only"

    v_only_cmd = control_suite.build_evaluate_cmd(
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
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=specs["fused_quant_v_only_cosine_brief"],
    )
    assert v_only_cmd[v_only_cmd.index("--fusion-rule") + 1] == "cosine"
    assert v_only_cmd[v_only_cmd.index("--kv-transport") + 1] == "v_only"

    attention_sparse_cmd = control_suite.build_evaluate_cmd(
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
        gate_search_file="gate.jsonl",
        gate_search_limit=12,
        spec=specs["fused_quant_k_only_attention_sparse_brief"],
    )
    assert attention_sparse_cmd[attention_sparse_cmd.index("--position-selection-ratio") + 1] == "0.5"
    assert attention_sparse_cmd[attention_sparse_cmd.index("--position-selection-metric") + 1] == "attention"


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


def test_best_metric_for_eval_handles_non_swept_rotalign_metrics() -> None:
    metrics = {
        "target_alone": 0.12,
        "text_to_text": 0.10,
        "rotalign_kv": 0.18,
        "rotalign_kv_bytes": 512.0,
    }

    best_metric, best_value = control_suite.best_metric_for_eval(
        metrics,
        methods=("rotalign",),
        gate_values=(0.15, 0.25),
        include_baselines=True,
    )

    assert best_metric == "rotalign_kv"
    assert best_value == 0.18


def test_best_metric_for_eval_handles_explicit_fused_protocol_metrics() -> None:
    metrics = {
        "target_alone": 0.12,
        "rotalign_fused": 0.19,
        "rotalign_fused_bytes": 256.0,
    }

    best_metric, best_value = control_suite.best_metric_for_eval(
        metrics,
        methods=("rotalign_fused",),
        gate_values=(0.15, 0.25),
    )

    assert best_metric == "rotalign_fused"
    assert best_value == 0.19


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


def test_load_existing_records_reads_jsonl_rows(tmp_path) -> None:
    jsonl_path = tmp_path / "suite_results.jsonl"
    jsonl_path.write_text(
        '{"checkpoint_tag":"ckpt","eval_name":"a"}\n'
        "\n"
        '{"checkpoint_tag":"ckpt","eval_name":"b"}\n',
        encoding="utf-8",
    )

    records = control_suite.load_existing_records(jsonl_path)

    assert [(record["checkpoint_tag"], record["eval_name"]) for record in records] == [
        ("ckpt", "a"),
        ("ckpt", "b"),
    ]


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
            calibration_specs=["ckpt"],
            eval_specs=["eval"],
            gate_search_file="gate.jsonl",
            gate_search_limit=9,
            reuse_checkpoints=True,
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
    assert plan["gate_search_file"] == "gate.jsonl"
    assert plan["gate_search_limit"] == 9
    assert plan["reuse_checkpoints"] is True
    assert not any((tmp_path / "checkpoints").iterdir())
