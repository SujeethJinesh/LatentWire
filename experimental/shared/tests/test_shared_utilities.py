import json
from pathlib import Path

import torch
import pytest

from experimental.shared.activation_dumper import load_tensor_packet, save_tensor_packet
from experimental.shared.boundary_inspector import LayerKind, boundaries_from_module_names
from experimental.shared.fp4_simulator import (
    gap_recovery_ratio,
    protect_positions,
    simulate_mxfp4_e2m1,
    simulate_symmetric_int,
)
from experimental.shared.hybrid_architecture_maps import build_map, write_maps
from experimental.shared.check_gate_packet import validate_gate_packet
from experimental.shared.hybrid_gate_evaluators import evaluate_ssq_lr_s1
from experimental.shared.hybrid_model_eligibility import (
    _architecture_hash,
    _local_cache_dir,
    _mac_trace_decision,
    _size_gb,
)
from experimental.shared.hybrid_trace_packet_builder import build_hbsm_packet, build_horn_packet, build_ssq_lr_packet
from experimental.shared.hybrid_trace_capture_manifest import build_capture_manifests
from experimental.shared.hybrid_trace_plan import write_trace_plan
from experimental.shared.sensitivity_metrics import kurtosis, rel_l2, spearman_rank_correlation


SSQ_BUCKETS = ("prefill_end", "2k_or_end", "8k_or_end", "final_minus_128")
TRACE_PLAN_HASHES = {
    "ssq_lr": "sha256:a05dab6ad3b821b91bd2e3c67340703bd7c7594e8d86b79051bfe763da17305b",
    "horn": "sha256:bde83105201b553340944f8c29bc94f8444f172e4fbe96d16951115208aa4c66",
    "hbsm": "sha256:5f9bea1f3a36920429f86f0c998ff92af6e8c8c99612d25cac46b0d3c6560acf",
}
GRANITE_TINY_REVISION = "791e0d3d28c86e106c9b6e0b4cecdee0375b6124"


def test_fp4_simulator_is_deterministic_and_shape_preserving() -> None:
    tensor = torch.linspace(-3, 3, steps=65).reshape(5, 13)
    first = simulate_mxfp4_e2m1(tensor, block_size=16)
    second = simulate_mxfp4_e2m1(tensor, block_size=16)

    assert first.dequantized.shape == tensor.shape
    assert torch.equal(first.dequantized, second.dequantized)
    assert first.format_name == "mxfp4_e2m1_sim"


def test_symmetric_int_quantization_and_protected_positions() -> None:
    tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    quantized = simulate_symmetric_int(tensor, bits=4, block_size=8).dequantized
    protected = protect_positions(tensor, quantized, protected_positions=[0], position_dim=1)

    assert torch.equal(protected[:, 0, :], tensor[:, 0, :])
    assert protected.shape == tensor.shape


def test_gap_recovery_ratio_gate_semantics() -> None:
    ratio = gap_recovery_ratio(bf16_score=1.0, uniform_score=2.0, protected_score=1.4)
    assert ratio == pytest.approx(0.4)


def test_boundary_inspector_finds_directional_hybrid_boundaries() -> None:
    boundaries = boundaries_from_module_names(
        ["layers.0.self_attn", "layers.1.mamba", "layers.2.mlp", "layers.3.attention"]
    )

    assert len(boundaries) == 1
    assert boundaries[0].left == LayerKind.ATTENTION
    assert boundaries[0].right == LayerKind.SSM
    assert boundaries[0].direction == "attention->ssm"


def test_sensitivity_metrics_are_well_formed() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0])
    candidate = torch.tensor([1.0, 2.0, 4.0])

    assert rel_l2(reference, candidate) > 0
    assert kurtosis(reference) > 0
    assert spearman_rank_correlation(reference, candidate) > 0.9


def test_tensor_packet_roundtrip(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    save_tensor_packet(
        packet,
        tensors={"layer/0 state": torch.ones(2, 3)},
        metadata={"model": "toy", "trace_count": 1},
    )

    tensors, metadata = load_tensor_packet(packet)

    assert metadata["model"] == "toy"
    assert torch.equal(tensors["layer_0_state"], torch.ones(2, 3))


def test_gate_packet_checker_accepts_valid_synthetic_packet(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "config.json").write_text('{"seed": 1}\n')
    (packet / "raw_rows.jsonl").write_text('{"layer": 0, "metric": 1.0}\n')
    (packet / "summary.json").write_text(
        "{"
        '"seed": 1, '
        '"surface": "synthetic_test", '
        '"decision": "SYNTHETIC_PASS", '
        '"rows": [{"layer": 0, "metric": 1.0}], '
        '"claim_boundary": ["synthetic-only"]'
        "}\n"
    )
    (packet / "decision.md").write_text("`SYNTHETIC_PASS`\n")

    report = validate_gate_packet(packet, expected_decision_prefix="SYNTHETIC")

    assert report["ok"]
    assert report["row_count"] == 1


def test_gate_packet_checker_rejects_missing_files(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()

    report = validate_gate_packet(packet)

    assert not report["ok"]
    assert "missing config.json" in report["errors"]


def test_gate_packet_checker_accepts_real_ssq_lr_contract(tmp_path: Path) -> None:
    packet = tmp_path / "real_ssq"
    packet.mkdir()
    (packet / "config.json").write_text(
        "{"
        '"model_id": "ibm-granite-4.0-h-tiny", '
        f'"model_revision": "{GRANITE_TINY_REVISION}", '
        f'"tokenizer_revision": "{GRANITE_TINY_REVISION}", '
        '"prompt_source": "fixed_manifest.json", '
        '"prompt_ids_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"seed_list": [1], '
        '"context_lengths": [128], '
        '"dtype": "bf16", '
        '"device": "mps", '
        '"architecture_map_hash": "sha256:bda8fd574ace7d968d82397f59ea6b9a702a077bbeab279a65b9dad7386a82c6", '
        f'"trace_plan_hash": "{TRACE_PLAN_HASHES["ssq_lr"]}", '
        '"command": "python run_gate.py"'
        "}\n"
    )
    rows = [
        {
            "model_id": "ibm-granite-4.0-h-tiny",
            "model_revision": GRANITE_TINY_REVISION,
            "prompt_id": f"p{prompt_index}",
            "layer": 0,
            "layer_kind": "mamba2",
            "position_bucket": bucket,
            "state_tensor_kind": "mamba2_recurrent_state",
            "state_shape": [1, 4],
            "max_abs": 2.0,
            "rms": 1.0,
            "std": 0.5,
            "kurtosis": 3.0,
            "outlier_mass": 0.1,
            "control_type": "bf16_no_quant",
        }
        for prompt_index in range(12)
        for bucket in SSQ_BUCKETS
    ]
    (packet / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    gate_status = str(evaluate_ssq_lr_s1(rows)["gate_status"])
    (packet / "summary.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "surface": "real_ssq_lr_s1",
                "decision": gate_status,
                "row_count": len(rows),
                "rows": rows,
                "claim_boundary": ["real model trace", "not GPU evidence"],
                "prompt_count": 12,
                "position_buckets": list(SSQ_BUCKETS),
                "ssm_layer_count": 1,
                "passing_layer_count": 0,
                "pass_fraction": 0.0,
                "selected_s1_ci_low": 1.0,
                "holm_p_min": 1.0,
                "max_abs_ratio_final_minus_128_vs_prefill_end": 1.0,
                "std_ratio_final_minus_128_vs_prefill_end": 1.0,
                "kurtosis_ratio_final_minus_128_vs_prefill_end": 1.0,
                **evaluate_ssq_lr_s1(rows),
            },
            sort_keys=True,
        )
        + "\n"
    )
    (packet / "summary.md").write_text("# Summary\n")
    (packet / "decision.md").write_text(f"`{gate_status}`\n")
    _attach_trace_plan_path_from_raw_rows(packet, tmp_path, "ssq_lr")

    report = validate_gate_packet(packet, mode="real", project="ssq_lr")

    assert report["ok"]
    assert report["mode"] == "real"


def test_gate_packet_checker_rejects_real_ssq_lr_incomplete_prompt_layer_matrix(tmp_path: Path) -> None:
    packet = tmp_path / "real_ssq_missing_matrix"
    packet.mkdir()
    config = _base_trace_metadata()
    (packet / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    rows = []
    for prompt_index in range(12):
        buckets = SSQ_BUCKETS if prompt_index != 11 else SSQ_BUCKETS[:-1]
        for bucket in buckets:
            rows.append(
                {
                    "model_id": "ibm-granite-4.0-h-tiny",
                    "model_revision": GRANITE_TINY_REVISION,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "state_shape": [1, 4],
                    "max_abs": 2.0,
                    "rms": 1.0,
                    "std": 0.5,
                    "kurtosis": 3.0,
                    "outlier_mass": 0.1,
                    "control_type": "bf16_no_quant",
                }
            )
    (packet / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    (packet / "summary.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "surface": "real_ssq_lr_s1",
                "decision": "CONTINUE_REAL_STATE_DUMPS",
                "row_count": len(rows),
                "rows": rows,
                "claim_boundary": ["real model trace", "not GPU evidence"],
                "prompt_count": 12,
                "position_buckets": list(SSQ_BUCKETS),
                "ssm_layer_count": 1,
                "passing_layer_count": 0,
                "pass_fraction": 0.0,
                "selected_s1_ci_low": 1.0,
                "holm_p_min": 1.0,
                "max_abs_ratio_final_minus_128_vs_prefill_end": 1.0,
                "std_ratio_final_minus_128_vs_prefill_end": 1.0,
                "kurtosis_ratio_final_minus_128_vs_prefill_end": 1.0,
                **evaluate_ssq_lr_s1(rows),
            },
            sort_keys=True,
        )
        + "\n"
    )
    (packet / "summary.md").write_text("# Summary\n")
    (packet / "decision.md").write_text("`CONTINUE_REAL_STATE_DUMPS`\n")

    report = validate_gate_packet(packet, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("every prompt/layer" in error for error in report["errors"])


def test_gate_packet_checker_rejects_promotable_resource_limited_real_packet(tmp_path: Path) -> None:
    packet = tmp_path / "real_ssq_resource_limited"
    packet.mkdir()
    config = _base_trace_metadata()
    config["resource_limit_note"] = "only two prompts fit locally"
    (packet / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    rows = [
        {
            "model_id": "ibm-granite-4.0-h-tiny",
            "model_revision": GRANITE_TINY_REVISION,
            "prompt_id": f"p{prompt_index}",
            "layer": 0,
            "layer_kind": "mamba2",
            "position_bucket": bucket,
            "state_tensor_kind": "mamba2_recurrent_state",
            "state_shape": [1, 4],
            "max_abs": 2.0,
            "rms": 1.0,
            "std": 0.5,
            "kurtosis": 3.0,
            "outlier_mass": 0.1,
            "control_type": "bf16_no_quant",
        }
        for prompt_index in range(2)
        for bucket in SSQ_BUCKETS
    ]
    (packet / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    (packet / "summary.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "surface": "real_ssq_lr_s1",
                "decision": "CONTINUE_REAL_STATE_DUMPS",
                "row_count": len(rows),
                "rows": rows,
                "claim_boundary": ["real model trace", "not GPU evidence"],
                "prompt_count": 2,
                "position_buckets": list(SSQ_BUCKETS),
                "ssm_layer_count": 1,
                "passing_layer_count": 0,
                "pass_fraction": 0.0,
                "selected_s1_ci_low": 1.0,
                "holm_p_min": 1.0,
                "max_abs_ratio_final_minus_128_vs_prefill_end": 1.0,
                "std_ratio_final_minus_128_vs_prefill_end": 1.0,
                "kurtosis_ratio_final_minus_128_vs_prefill_end": 1.0,
                **evaluate_ssq_lr_s1(rows),
            },
            sort_keys=True,
        )
        + "\n"
    )
    (packet / "summary.md").write_text("# Summary\n")
    (packet / "decision.md").write_text("`CONTINUE_REAL_STATE_DUMPS`\n")

    report = validate_gate_packet(packet, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("resource-limited real packet" in error for error in report["errors"])


def test_gate_packet_checker_rejects_real_ssq_lr_without_all_position_buckets(tmp_path: Path) -> None:
    packet = tmp_path / "real_ssq_missing_bucket"
    packet.mkdir()
    config = _base_trace_metadata()
    config["device"] = "mps"
    (packet / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    rows = [
        {
            "model_id": "ibm-granite-4.0-h-tiny",
            "model_revision": GRANITE_TINY_REVISION,
            "prompt_id": f"p{prompt_index}",
            "layer": 0,
            "layer_kind": "mamba2",
            "position_bucket": "prefill_end",
            "state_tensor_kind": "mamba2_recurrent_state",
            "state_shape": [1, 4],
            "max_abs": 2.0,
            "rms": 1.0,
            "std": 0.5,
            "kurtosis": 3.0,
            "outlier_mass": 0.1,
            "control_type": "bf16_no_quant",
        }
        for prompt_index in range(12)
    ]
    (packet / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )
    (packet / "summary.json").write_text(
        json.dumps(
            {
                "seed": 1,
                "surface": "real_ssq_lr_s1",
                "decision": "CONTINUE_REAL_STATE_DUMPS",
                "row_count": len(rows),
                "rows": rows,
                "claim_boundary": ["real model trace", "not GPU evidence"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    (packet / "summary.md").write_text("# Summary\n")
    (packet / "decision.md").write_text("`CONTINUE_REAL_STATE_DUMPS`\n")

    report = validate_gate_packet(packet, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("missing position buckets" in error for error in report["errors"])


def test_gate_packet_checker_rejects_real_packet_without_project_controls(tmp_path: Path) -> None:
    packet = tmp_path / "real_horn"
    packet.mkdir()
    (packet / "config.json").write_text(
        "{"
        '"model_id": "ibm-granite-4.0-h-tiny", '
        f'"model_revision": "{GRANITE_TINY_REVISION}", '
        f'"tokenizer_revision": "{GRANITE_TINY_REVISION}", '
        '"prompt_source": "fixed_manifest.json", '
        '"prompt_ids_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"seed_list": [1], '
        '"context_lengths": [128], '
        '"dtype": "bf16", '
        '"device": "mps", '
        '"architecture_map_hash": "sha256:bda8fd574ace7d968d82397f59ea6b9a702a077bbeab279a65b9dad7386a82c6", '
        '"trace_plan_hash": "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", '
        '"command": "python run_gate.py"'
        "}\n"
    )
    (packet / "raw_rows.jsonl").write_text(
        "{"
        '"model_id": "ibm-granite-4.0-h-tiny", '
        '"prompt_id": "p0", '
        '"layer_left": 0, '
        '"layer_right": 1, '
        '"direction": "attention->ssm", '
        '"matched_boundary_direction": "attention->ssm", '
        '"boundary_index": 0, '
        '"pre_norm_position": "post_norm", '
        '"post_norm_position": "pre_norm", '
        '"max_abs": 2.0, '
        '"rms": 1.0, '
        '"kurtosis": 3.0, '
        '"control_type": "boundary"'
        "}\n"
    )
    (packet / "summary.json").write_text(
        "{"
        '"seed": 1, '
        '"surface": "real_horn_h1", '
        '"decision": "CONTINUE_REAL_BOUNDARY_DUMPS", '
        '"row_count": 1, '
        '"rows": [{"boundary_index": 0}], '
        '"claim_boundary": ["real model trace", "not GPU evidence"]'
        "}\n"
    )
    (packet / "summary.md").write_text("# Summary\n")
    (packet / "decision.md").write_text("`CONTINUE_REAL_BOUNDARY_DUMPS`\n")

    report = validate_gate_packet(packet, mode="real", project="horn")

    assert not report["ok"]
    assert "missing required controls: non_boundary, permuted_direction" in report["errors"]


def test_gate_packet_checker_rejects_unpaired_horn_permuted_prompt(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn_bad_pair"
    metadata = _base_trace_metadata("horn")
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for stem, layer_left, layer_right, direction, boundary_index, control_type, prompt_id in [
            ("boundary_attn_ssm", 0, 1, "attention->ssm", 0, "boundary", f"p{prompt_index}"),
            ("boundary_ssm_attn", 2, 3, "ssm->attention", 1, "boundary", f"p{prompt_index}"),
            ("non_boundary_attn_ssm", 4, 5, "attention->ssm", 2, "non_boundary", f"p{prompt_index}"),
            ("non_boundary_ssm_attn", 6, 7, "ssm->attention", 3, "non_boundary", f"p{prompt_index}"),
            ("permuted_attn_ssm", 0, 1, "ssm->attention", 0, "permuted_direction", f"other{prompt_index}"),
            ("permuted_ssm_attn", 2, 3, "attention->ssm", 1, "permuted_direction", f"other{prompt_index}"),
        ]:
            actual_direction = "ssm->ssm" if control_type == "non_boundary" else direction
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": prompt_id,
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": actual_direction,
                    "matched_boundary_direction": direction,
                    "boundary_index": boundary_index,
                    "pre_norm_position": "post_norm",
                    "post_norm_position": "pre_norm",
                    "control_type": control_type,
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["horn_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_horn_packet(tensor_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert not report["ok"]
    assert any("must match an observed boundary tuple" in error for error in report["errors"])


def test_hybrid_architecture_map_uses_explicit_layer_types(tmp_path: Path) -> None:
    config = tmp_path / "toy.config.json"
    config.write_text(
        "{"
        '"architectures": ["ToyHybrid"], '
        '"model_type": "toyhybrid", '
        '"hidden_size": 16, '
        '"layer_types": ["mamba", "attention", "mamba", "mamba"]'
        "}\n"
    )

    row = build_map(config)

    assert row["boundary_count"] == 2
    assert row["direction_counts"] == {"attention->ssm": 1, "ssm->attention": 1}
    assert row["boundaries"][0]["left_layer"] == 0
    assert row["boundaries"][0]["right_layer"] == 1


def test_hybrid_architecture_map_packet_contains_controls(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    output_dir = tmp_path / "maps"
    config_dir.mkdir()
    (config_dir / "toy.config.json").write_text(
        "{"
        '"architectures": ["ToyHybrid"], '
        '"model_type": "toyhybrid", '
        '"hidden_size": 16, '
        '"layer_types": ["mamba", "attention", "mamba", "mamba"]'
        "}\n"
    )

    write_maps(config_dir=config_dir, output_dir=output_dir)

    rows = [(output_dir / "raw_rows.jsonl").read_text()]
    assert "permuted_direction_control" in rows[0]
    assert "non_boundary_control" in rows[0]
    assert "CONFIG_ONLY_READY_FOR_TRACE_PACKET_PROVENANCE" in (output_dir / "decision.md").read_text()


def test_hybrid_trace_plan_enumerates_project_specific_rows(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    maps_dir = tmp_path / "maps"
    output_dir = tmp_path / "trace_plan"
    prompts = tmp_path / "prompts.jsonl"
    config_dir.mkdir()
    (config_dir / "toy.config.json").write_text(
        "{"
        '"architectures": ["ToyHybrid"], '
        '"model_type": "toyhybrid", '
        '"hidden_size": 16, '
        '"layer_types": ["mamba", "attention", "mamba", "mamba"]'
        "}\n"
    )
    prompts.write_text(
        '{"prompt_id": "p0", "prompt": "one"}\n'
        '{"prompt_id": "p1", "prompt": "two"}\n'
    )
    write_maps(config_dir=config_dir, output_dir=maps_dir)

    summary = write_trace_plan(
        output_dir=output_dir,
        prompts_path=prompts,
        architecture_maps_path=maps_dir / "architecture_maps.json",
    )

    assert summary["decision"] == "TRACE_PLAN_READY_NOT_MODEL_EVIDENCE"
    assert summary["row_counts"] == {"hbsm": 14, "horn": 12, "ssq_lr": 24}
    horn_rows = [
        json.loads(line)
        for line in (output_dir / "horn_trace_plan.jsonl").read_text().splitlines()
    ]
    assert {row["control_type"] for row in horn_rows} == {
        "boundary",
        "non_boundary",
        "permuted_direction",
    }
    assert {
        row["matched_boundary_direction"]
        for row in horn_rows
        if row["control_type"] == "non_boundary"
    } == {"attention->ssm", "ssm->attention"}
    assert "not model evidence" in (output_dir / "decision.md").read_text()


def test_hybrid_trace_plan_uses_architecture_specific_ssq_state_labels(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    maps_dir = tmp_path / "maps"
    output_dir = tmp_path / "trace_plan"
    prompts = tmp_path / "prompts.jsonl"
    config_dir.mkdir()
    (config_dir / "qwen3-next-80b-a3b-instruct.config.json").write_text(
        "{"
        '"architectures": ["Qwen3NextForCausalLM"], '
        '"model_type": "qwen3_next", '
        '"hidden_size": 16, '
        '"num_hidden_layers": 5, '
        '"full_attention_interval": 3'
        "}\n"
    )
    prompts.write_text('{"prompt_id": "p0", "prompt": "one"}\n')
    write_maps(config_dir=config_dir, output_dir=maps_dir)

    write_trace_plan(
        output_dir=output_dir,
        prompts_path=prompts,
        architecture_maps_path=maps_dir / "architecture_maps.json",
    )

    ssq_rows = [
        json.loads(line)
        for line in (output_dir / "ssq_lr_trace_plan.jsonl").read_text().splitlines()
    ]
    assert ssq_rows
    assert {row["layer_kind"] for row in ssq_rows} == {"ssm"}
    assert {row["state_tensor_kind"] for row in ssq_rows} == {"recurrent_ssm_state"}


def test_hybrid_capture_manifest_templates_are_generated_from_trace_plan(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    maps_dir = tmp_path / "maps"
    trace_plan_dir = tmp_path / "trace_plan"
    manifest_dir = tmp_path / "capture_manifests"
    prompts = tmp_path / "prompts.jsonl"
    config_dir.mkdir()
    (config_dir / "toy.config.json").write_text(
        "{"
        '"architectures": ["ToyHybrid"], '
        '"model_type": "toyhybrid", '
        '"hidden_size": 16, '
        '"layer_types": ["mamba", "attention", "mamba", "mamba"]'
        "}\n"
    )
    prompts.write_text(
        '{"prompt_id": "p0", "prompt": "one"}\n'
        '{"prompt_id": "p1", "prompt": "two"}\n'
    )
    write_maps(config_dir=config_dir, output_dir=maps_dir)
    write_trace_plan(
        output_dir=trace_plan_dir,
        prompts_path=prompts,
        architecture_maps_path=maps_dir / "architecture_maps.json",
    )

    summary = build_capture_manifests(trace_plan_dir=trace_plan_dir, output_dir=manifest_dir)

    assert summary["decision"] == "CAPTURE_MANIFEST_READY_NOT_MODEL_EVIDENCE"
    assert summary["counts"]["ssq_lr"]["toy"] == 24
    ssq_template = json.loads((manifest_dir / "ssq_lr__toy__metadata_template.json").read_text())
    horn_template = json.loads((manifest_dir / "horn__toy__metadata_template.json").read_text())
    hbsm_template = json.loads((manifest_dir / "hbsm__toy__row_packet_template.json").read_text())
    assert ssq_template["_template_only"] is True
    assert len(ssq_template["ssq_lr_entries"]) == 24
    assert ssq_template["trace_plan_hash"].startswith("sha256:")
    assert len(horn_template["horn_entries"]) == 12
    assert "hbsm_entries" not in hbsm_template
    assert len(hbsm_template["hbsm_entry_templates"]) == 14
    assert "not GPU evidence" in (manifest_dir / "summary.md").read_text()


def test_hybrid_model_eligibility_helpers_are_repo_local(tmp_path: Path) -> None:
    cache = _local_cache_dir(tmp_path / "hf_home", "owner/model-name")

    assert cache == tmp_path / "hf_home/hub/models--owner--model-name"
    assert _size_gb(1024**3) == 1.0


def test_hybrid_model_eligibility_matches_hf_and_fp8_slugs() -> None:
    small_hash = _architecture_hash("ibm-granite/granite-4.0-h-small")
    fp8_hash = _architecture_hash("ibm-granite/granite-4.0-h-small-FP8")

    assert small_hash
    assert fp8_hash == small_hash


def test_hybrid_model_eligibility_preserves_gpu_recommendation_when_not_cached() -> None:
    assert (
        _mac_trace_decision(
            weights_cached=False,
            estimated_weight_gb=60.0,
            requires_mamba_ssm=True,
            mamba_ssm_installed=False,
        )
        == "GPU_RECOMMENDED_SIZE_NOT_CACHED"
    )
    assert (
        _mac_trace_decision(
            weights_cached=False,
            estimated_weight_gb=8.0,
            requires_mamba_ssm=False,
            mamba_ssm_installed=True,
        )
        == "BLOCKED_NOT_CACHED"
    )


def _base_trace_metadata(project: str = "ssq_lr") -> dict[str, object]:
    return {
        "model_id": "ibm-granite-4.0-h-tiny",
        "model_revision": GRANITE_TINY_REVISION,
        "tokenizer_revision": GRANITE_TINY_REVISION,
        "prompt_source": "fixed_manifest.json",
        "prompt_ids_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "seed_list": [1],
        "context_lengths": [16],
        "dtype": "bf16",
        "device": "cpu",
        "command": "python dump.py",
        "architecture_map_hash": "sha256:bda8fd574ace7d968d82397f59ea6b9a702a077bbeab279a65b9dad7386a82c6",
        "trace_plan_hash": TRACE_PLAN_HASHES[project],
    }


def _attach_trace_plan_path_from_raw_rows(packet_dir: Path, tmp_path: Path, project: str) -> None:
    rows = [json.loads(line) for line in (packet_dir / "raw_rows.jsonl").read_text().splitlines()]
    trace_plan = tmp_path / f"{packet_dir.name}_{project}_trace_plan.jsonl"
    trace_plan.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    config = json.loads((packet_dir / "config.json").read_text(encoding="utf-8"))
    config["trace_plan_path"] = str(trace_plan)
    (packet_dir / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")


def test_ssq_lr_packet_builder_outputs_checker_compatible_real_packet(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq"
    metadata = _base_trace_metadata()
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(
        tensor_packet,
        tensors=tensors,
        metadata=metadata,
    )

    build_ssq_lr_packet(tensor_packet, output_dir)
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "ssq_lr")
    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert report["ok"]
    assert report["row_count"] == 48


def test_packet_builder_rejects_unfilled_capture_templates(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_template"
    metadata = _base_trace_metadata()
    metadata["_template_only"] = True
    metadata["model_revision"] = "TO_FILL_BEFORE_CAPTURE"
    metadata["ssq_lr_entries"] = [
        {
            "tensor": "state",
            "prompt_id": "p0",
            "layer": 0,
            "layer_kind": "mamba2",
            "position_bucket": "prefill_end",
            "state_tensor_kind": "mamba2_recurrent_state",
            "control_type": "bf16_no_quant",
        }
    ]
    save_tensor_packet(tensor_packet, tensors={"state": torch.ones(2, 4)}, metadata=metadata)

    with pytest.raises(ValueError, match="capture template"):
        build_ssq_lr_packet(tensor_packet, output_dir)


def test_packet_builder_marks_resource_limited_packets_non_promotable(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_resource_limited"
    metadata = _base_trace_metadata()
    metadata["resource_limit_note"] = "two-prompt local smoke before full trace packet"
    entries = []
    tensors = {}
    for prompt_index in range(2):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.full((2, 4), float(prompt_index + len(bucket)))
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_ssq_lr_packet(tensor_packet, output_dir)
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "ssq_lr")
    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert summary["decision"].startswith("RESOURCE_LIMITED_NOT_PROMOTABLE_")
    assert report["ok"]


def test_gate_packet_checker_rejects_unknown_architecture_map_hash(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_bad_arch_hash"
    metadata = _base_trace_metadata()
    metadata["architecture_map_hash"] = "sha256:" + ("d" * 64)
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_ssq_lr_packet(tensor_packet, output_dir)
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "ssq_lr")
    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("architecture_map_hash must match" in error for error in report["errors"])


def test_gate_packet_checker_requires_trace_plan_hash_for_real_packets(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_missing_trace_plan_hash"
    metadata = _base_trace_metadata()
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)
    build_ssq_lr_packet(tensor_packet, output_dir)
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    config.pop("trace_plan_hash")
    (output_dir / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("trace_plan_hash" in error for error in report["errors"])


def test_gate_packet_checker_requires_trace_plan_path_for_real_packets(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_missing_trace_plan_path"
    metadata = _base_trace_metadata()
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)
    build_ssq_lr_packet(tensor_packet, output_dir)

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("trace_plan_path" in error for error in report["errors"])


def test_gate_packet_checker_rejects_wrong_project_trace_plan_hash(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_wrong_trace_plan_hash"
    metadata = _base_trace_metadata()
    metadata["trace_plan_hash"] = TRACE_PLAN_HASHES["horn"]
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)
    build_ssq_lr_packet(tensor_packet, output_dir)

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("trace_plan_hash must match shared trace-plan hash" in error for error in report["errors"])


def test_packet_builder_resolves_sanitized_tensor_names(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_sanitized"
    metadata = _base_trace_metadata()
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"layers/0 state {bucket} p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_ssq_lr_packet(tensor_packet, output_dir)
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "ssq_lr")
    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert report["ok"]
    assert report["row_count"] == 48


def test_packet_builder_canonicalizes_served_hf_model_id(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_hf_alias"
    metadata = _base_trace_metadata()
    metadata["model_id"] = "ibm-granite/granite-4.0-h-tiny"
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_ssq_lr_packet(tensor_packet, output_dir)
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "ssq_lr")
    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output_dir / "raw_rows.jsonl").read_text().splitlines()]

    assert report["ok"]
    assert config["model_id"] == "ibm-granite-4.0-h-tiny"
    assert config["served_model_id"] == "ibm-granite/granite-4.0-h-tiny"
    assert {row["model_id"] for row in rows} == {"ibm-granite-4.0-h-tiny"}


def test_gate_packet_checker_rejects_unregistered_model_revision(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_bad_revision"
    metadata = _base_trace_metadata()
    metadata["model_revision"] = "0" * 40
    metadata["tokenizer_revision"] = GRANITE_TINY_REVISION
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_ssq_lr_packet(tensor_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("model_revision must match" in error for error in report["errors"])


def test_gate_packet_checker_rejects_decision_that_disagrees_with_evaluator(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_bad_decision"
    metadata = _base_trace_metadata()
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)
    build_ssq_lr_packet(tensor_packet, output_dir)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    summary["decision"] = "PROMOTE_UNVERIFIED"
    (output_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
    (output_dir / "decision.md").write_text("`PROMOTE_UNVERIFIED`\n")

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("summary decision must equal recomputed gate decision" in error for error in report["errors"])


def test_gate_packet_checker_rejects_rows_outside_cited_trace_plan(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_bad_trace_plan_row"
    metadata = _base_trace_metadata()
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for bucket in SSQ_BUCKETS:
            tensor_name = f"state_layer_0_{bucket}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "layer_kind": "mamba2",
                    "position_bucket": bucket,
                    "state_tensor_kind": "mamba2_recurrent_state",
                    "control_type": "bf16_no_quant",
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["ssq_lr_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)
    build_ssq_lr_packet(tensor_packet, output_dir)

    rows = [json.loads(line) for line in (output_dir / "raw_rows.jsonl").read_text().splitlines()]
    trace_plan = tmp_path / "ssq_trace_plan.jsonl"
    trace_plan.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    config["trace_plan_path"] = str(trace_plan)
    (output_dir / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")

    rows[0]["prompt_id"] = "unplanned_prompt"
    (output_dir / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("outside the frozen trace plan" in error for error in report["errors"])


def test_horn_packet_builder_outputs_required_controls(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn"
    metadata = _base_trace_metadata("horn")
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for stem, layer_left, layer_right, direction, boundary_index, control_type in [
            ("boundary_attn_ssm", 0, 1, "attention->ssm", 0, "boundary"),
            ("boundary_ssm_attn", 2, 3, "ssm->attention", 1, "boundary"),
            ("non_boundary_attn_ssm", 4, 5, "attention->ssm", 2, "non_boundary"),
            ("non_boundary_ssm_attn", 6, 7, "ssm->attention", 3, "non_boundary"),
            ("permuted_attn_ssm", 0, 1, "ssm->attention", 0, "permuted_direction"),
            ("permuted_ssm_attn", 2, 3, "attention->ssm", 1, "permuted_direction"),
        ]:
            actual_direction = "ssm->ssm" if control_type == "non_boundary" else direction
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": actual_direction,
                    "matched_boundary_direction": direction,
                    "boundary_index": boundary_index,
                    "pre_norm_position": "post_norm",
                    "post_norm_position": "pre_norm",
                    "control_type": control_type,
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
        tensors[f"permuted_attn_ssm_p{prompt_index}"] = tensors[f"boundary_attn_ssm_p{prompt_index}"].clone()
        tensors[f"permuted_ssm_attn_p{prompt_index}"] = tensors[f"boundary_ssm_attn_p{prompt_index}"].clone()
    metadata["horn_entries"] = entries
    save_tensor_packet(
        tensor_packet,
        tensors=tensors,
        metadata=metadata,
    )

    build_horn_packet(tensor_packet, output_dir)
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "horn")
    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert report["ok"]
    assert report["row_count"] == 72


def test_horn_packet_builder_uses_tensor_alias_for_permuted_rows(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn_alias"
    metadata = _base_trace_metadata("horn")
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for stem, layer_left, layer_right, direction, boundary_index, control_type in [
            ("boundary_attn_ssm", 0, 1, "attention->ssm", 0, "boundary"),
            ("boundary_ssm_attn", 2, 3, "ssm->attention", 1, "boundary"),
            ("non_boundary_attn_ssm", 4, 5, "attention->ssm", 2, "non_boundary"),
            ("non_boundary_ssm_attn", 6, 7, "ssm->attention", 3, "non_boundary"),
            ("permuted_attn_ssm", 0, 1, "ssm->attention", 0, "permuted_direction"),
            ("permuted_ssm_attn", 2, 3, "attention->ssm", 1, "permuted_direction"),
        ]:
            actual_direction = "ssm->ssm" if control_type == "non_boundary" else direction
            tensor_name = f"{stem}_p{prompt_index}"
            entry = {
                "tensor": tensor_name,
                "prompt_id": f"p{prompt_index}",
                "layer_left": layer_left,
                "layer_right": layer_right,
                "direction": actual_direction,
                "matched_boundary_direction": direction,
                "boundary_index": boundary_index,
                "pre_norm_position": "post_norm",
                "post_norm_position": "pre_norm",
                "control_type": control_type,
            }
            if control_type == "permuted_direction":
                source = "boundary_attn_ssm" if boundary_index == 0 else "boundary_ssm_attn"
                entry["tensor_alias_of"] = f"{source}_p{prompt_index}"
            else:
                tensors[tensor_name] = torch.randn(2, 4)
            entries.append(entry)
    metadata["horn_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_horn_packet(tensor_packet, output_dir)
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "horn")
    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert report["ok"]
    assert report["row_count"] == 72


def test_gate_packet_checker_rejects_horn_permuted_without_actual_direction_flip(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn_fake_flip"
    metadata = _base_trace_metadata("horn")
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for stem, layer_left, layer_right, direction, boundary_index, control_type in [
            ("boundary_attn_ssm", 0, 1, "attention->ssm", 0, "boundary"),
            ("boundary_ssm_attn", 2, 3, "ssm->attention", 1, "boundary"),
            ("non_boundary_attn_ssm", 4, 5, "attention->ssm", 2, "non_boundary"),
            ("non_boundary_ssm_attn", 6, 7, "ssm->attention", 3, "non_boundary"),
            ("permuted_attn_ssm", 0, 1, "ssm->attention", 0, "permuted_direction"),
            ("permuted_ssm_attn", 2, 3, "attention->ssm", 1, "permuted_direction"),
        ]:
            actual_direction = "ssm->ssm" if control_type == "non_boundary" else direction
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": actual_direction,
                    "matched_boundary_direction": direction,
                    "boundary_index": boundary_index,
                    "pre_norm_position": "post_norm",
                    "post_norm_position": "pre_norm",
                    "control_type": control_type,
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
        tensors[f"permuted_attn_ssm_p{prompt_index}"] = tensors[f"boundary_attn_ssm_p{prompt_index}"].clone()
        tensors[f"permuted_ssm_attn_p{prompt_index}"] = tensors[f"boundary_ssm_attn_p{prompt_index}"].clone()
    metadata["horn_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)
    build_horn_packet(tensor_packet, output_dir)

    raw_rows_path = output_dir / "raw_rows.jsonl"
    rows = [json.loads(line) for line in raw_rows_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["control_type"] == "permuted_direction" and row["boundary_index"] == 0:
            row["direction"] = "attention->ssm"
            row["matched_boundary_direction"] = "ssm->attention"
            break
    raw_rows_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert not report["ok"]
    assert any("must flip the observed boundary direction" in error for error in report["errors"])
    assert any("matched_boundary_direction must equal flipped direction" in error for error in report["errors"])


def test_gate_packet_checker_rejects_horn_unpaired_non_boundary_prompt(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn_bad_non_boundary_pair"
    metadata = _base_trace_metadata("horn")
    entries = []
    tensors = {}
    for prompt_index in range(12):
        non_boundary_specs = [
            ("non_boundary_attn_ssm", 4, 5, "attention->ssm", 2, "non_boundary"),
        ]
        if prompt_index != 0:
            non_boundary_specs.append(
                ("non_boundary_ssm_attn", 6, 7, "ssm->attention", 3, "non_boundary")
            )
        for stem, layer_left, layer_right, direction, boundary_index, control_type in [
            ("boundary_attn_ssm", 0, 1, "attention->ssm", 0, "boundary"),
            ("boundary_ssm_attn", 2, 3, "ssm->attention", 1, "boundary"),
            *non_boundary_specs,
            ("permuted_attn_ssm", 0, 1, "ssm->attention", 0, "permuted_direction"),
            ("permuted_ssm_attn", 2, 3, "attention->ssm", 1, "permuted_direction"),
        ]:
            actual_direction = "ssm->ssm" if control_type == "non_boundary" else direction
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": actual_direction,
                    "matched_boundary_direction": direction,
                    "boundary_index": boundary_index,
                    "pre_norm_position": "post_norm",
                    "post_norm_position": "pre_norm",
                    "control_type": control_type,
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
        tensors[f"permuted_attn_ssm_p{prompt_index}"] = tensors[f"boundary_attn_ssm_p{prompt_index}"].clone()
        tensors[f"permuted_ssm_attn_p{prompt_index}"] = tensors[f"boundary_ssm_attn_p{prompt_index}"].clone()
    metadata["horn_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_horn_packet(tensor_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert not report["ok"]
    assert any("non_boundary controls must match both boundary directions for every prompt" in error for error in report["errors"])


def test_gate_packet_checker_rejects_horn_missing_permuted_pair(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn_missing_permuted_pair"
    metadata = _base_trace_metadata("horn")
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for stem, layer_left, layer_right, direction, boundary_index, control_type in [
            ("boundary_attn_ssm", 0, 1, "attention->ssm", 0, "boundary"),
            ("boundary_ssm_attn", 2, 3, "ssm->attention", 1, "boundary"),
            ("non_boundary_attn_ssm", 4, 5, "attention->ssm", 2, "non_boundary"),
            ("non_boundary_ssm_attn", 6, 7, "ssm->attention", 3, "non_boundary"),
            ("permuted_attn_ssm", 0, 1, "ssm->attention", 0, "permuted_direction"),
            ("permuted_ssm_attn", 2, 3, "attention->ssm", 1, "permuted_direction"),
        ]:
            if prompt_index == 0 and control_type == "permuted_direction" and boundary_index == 0:
                continue
            actual_direction = "ssm->ssm" if control_type == "non_boundary" else direction
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": actual_direction,
                    "matched_boundary_direction": direction,
                    "boundary_index": boundary_index,
                    "pre_norm_position": "post_norm",
                    "post_norm_position": "pre_norm",
                    "control_type": control_type,
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
        tensors[f"permuted_ssm_attn_p{prompt_index}"] = tensors[f"boundary_ssm_attn_p{prompt_index}"].clone()
        if prompt_index != 0:
            tensors[f"permuted_attn_ssm_p{prompt_index}"] = tensors[f"boundary_attn_ssm_p{prompt_index}"].clone()
    metadata["horn_entries"] = entries
    save_tensor_packet(tensor_packet, tensors=tensors, metadata=metadata)

    build_horn_packet(tensor_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert not report["ok"]
    assert any("paired permuted_direction" in error for error in report["errors"])


def test_gate_packet_checker_rejects_horn_permuted_direction_without_matching_boundary(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn_bad"
    metadata = _base_trace_metadata("horn")
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for stem, layer_left, layer_right, direction, boundary_index, control_type in [
            ("boundary_attn_ssm", 0, 1, "attention->ssm", 0, "boundary"),
            ("boundary_ssm_attn", 2, 3, "ssm->attention", 1, "boundary"),
            ("non_boundary_attn_ssm", 4, 5, "attention->ssm", 2, "non_boundary"),
            ("non_boundary_ssm_attn", 6, 7, "ssm->attention", 3, "non_boundary"),
            ("permuted_unmatched", 8, 9, "attention->ssm", 9, "permuted_direction"),
        ]:
            actual_direction = "ssm->ssm" if control_type == "non_boundary" else direction
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": actual_direction,
                    "matched_boundary_direction": direction,
                    "boundary_index": boundary_index,
                    "pre_norm_position": "post_norm",
                    "post_norm_position": "pre_norm",
                    "control_type": control_type,
                }
            )
            tensors[tensor_name] = torch.randn(2, 4)
    metadata["horn_entries"] = entries
    save_tensor_packet(
        tensor_packet,
        tensors=tensors,
        metadata=metadata,
    )

    build_horn_packet(tensor_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert not report["ok"]
    assert any("must match an observed boundary tuple" in error for error in report["errors"])


def test_gate_packet_checker_rejects_horn_permuted_direction_with_independent_metrics(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn_bad_metrics"
    metadata = _base_trace_metadata("horn")
    entries = []
    tensors = {}
    for prompt_index in range(12):
        for stem, layer_left, layer_right, direction, boundary_index, control_type in [
            ("boundary_attn_ssm", 0, 1, "attention->ssm", 0, "boundary"),
            ("boundary_ssm_attn", 2, 3, "ssm->attention", 1, "boundary"),
            ("non_boundary_attn_ssm", 4, 5, "attention->ssm", 2, "non_boundary"),
            ("non_boundary_ssm_attn", 6, 7, "ssm->attention", 3, "non_boundary"),
            ("permuted_attn_ssm", 0, 1, "ssm->attention", 0, "permuted_direction"),
            ("permuted_ssm_attn", 2, 3, "attention->ssm", 1, "permuted_direction"),
        ]:
            actual_direction = "ssm->ssm" if control_type == "non_boundary" else direction
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": actual_direction,
                    "matched_boundary_direction": direction,
                    "boundary_index": boundary_index,
                    "pre_norm_position": "post_norm",
                    "post_norm_position": "pre_norm",
                    "control_type": control_type,
                }
            )
            offset = 100.0 if control_type == "permuted_direction" else 0.0
            tensors[tensor_name] = torch.randn(2, 4) + offset
    metadata["horn_entries"] = entries
    save_tensor_packet(
        tensor_packet,
        tensors=tensors,
        metadata=metadata,
    )

    build_horn_packet(tensor_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="horn")

    assert not report["ok"]
    assert any("reuse the observed boundary metrics" in error for error in report["errors"])


def test_hbsm_packet_builder_outputs_required_controls(tmp_path: Path) -> None:
    row_packet = tmp_path / "hbsm_rows.json"
    output_dir = tmp_path / "hbsm"
    metadata = _base_trace_metadata("hbsm")
    primary_entries = []
    for index in range(60):
        boundary = index < 30
        primary_entries.append(
            {
                "prompt_id": f"p{index % 30}",
                "layer": index,
                "boundary_flag": boundary,
                "precision_perturbation": "mxfp4_e2m1",
                "kl_or_nll_drift": float(100 - index) * 0.01,
                "cheap_predictor": float(100 - index),
                "parameter_count": 1024 + index,
                "weight_norm": 0.5 + index,
                "top_decile_flag": boundary and index < 6,
                "random_top_decile": index in {0, 1, 2, 30, 31, 32},
                "train_test_split": "train" if index % 2 == 0 else "test",
                "control_type": "boundary_only",
            }
        )
    control_entries = [
        {
            "prompt_id": f"control_{control}",
            "layer": 100 + index,
            "boundary_flag": False,
            "precision_perturbation": "mxfp4_e2m1",
            "kl_or_nll_drift": 0.0,
            "cheap_predictor": 0.0,
            "parameter_count": 1024,
            "weight_norm": 0.5,
            "top_decile_flag": False,
            "random_top_decile": False,
            "train_test_split": "train",
            "control_type": control,
        }
        for index, control in enumerate(["perturbation_off", "random_flags", "layer_index", "parameter_count_norm"])
    ]
    control_entries.extend(
        [
            {
                "prompt_id": "control_kl_lens_rank",
                "layer": 200,
                "boundary_flag": False,
                "precision_perturbation": "mxfp4_e2m1",
                "kl_or_nll_drift": 0.0,
                "cheap_predictor": 0.0,
                "parameter_count": 1024,
                "weight_norm": 0.5,
                "top_decile_flag": False,
                "random_top_decile": False,
                "train_test_split": "train",
                "control_type": "kl_lens_rank",
            },
            {
                "prompt_id": "control_activation_outlier",
                "layer": 201,
                "boundary_flag": False,
                "precision_perturbation": "mxfp4_e2m1",
                "kl_or_nll_drift": 0.0,
                "cheap_predictor": 0.0,
                "parameter_count": 1024,
                "weight_norm": 0.5,
                "top_decile_flag": False,
                "random_top_decile": False,
                "train_test_split": "train",
                "control_type": "activation_outlier",
            },
        ]
    )
    row_packet.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "hbsm_entries": primary_entries + control_entries,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    build_hbsm_packet(row_packet, output_dir)
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "hbsm")
    report = validate_gate_packet(output_dir, mode="real", project="hbsm")

    assert report["ok"]
    assert report["row_count"] == 66


def test_hbsm_checker_rejects_top_decile_flags_that_disagree_with_drift(tmp_path: Path) -> None:
    row_packet = tmp_path / "hbsm_rows.json"
    output_dir = tmp_path / "hbsm_mismatched_top"
    metadata = _base_trace_metadata("hbsm")
    primary_entries = []
    for index in range(60):
        boundary = index < 30
        primary_entries.append(
            {
                "prompt_id": f"p{index % 30}",
                "layer": index,
                "boundary_flag": boundary,
                "precision_perturbation": "mxfp4_e2m1",
                "kl_or_nll_drift": float(index) * 0.01,
                "cheap_predictor": float(index + 1),
                "parameter_count": 1024 + index,
                "weight_norm": 0.5 + index,
                "top_decile_flag": boundary and index < 6,
                "random_top_decile": index in {0, 1, 2, 30, 31, 32},
                "train_test_split": "train" if index % 2 == 0 else "test",
                "control_type": "boundary_only",
            }
        )
    controls = [
        "perturbation_off",
        "random_flags",
        "layer_index",
        "parameter_count_norm",
        "kl_lens_rank",
        "activation_outlier",
    ]
    control_entries = [
        {
            "prompt_id": f"control_{control}",
            "layer": 100 + index,
            "boundary_flag": False,
            "precision_perturbation": "mxfp4_e2m1",
            "kl_or_nll_drift": 0.0,
            "cheap_predictor": 0.0,
            "parameter_count": 1024,
            "weight_norm": 0.5,
            "top_decile_flag": False,
            "random_top_decile": False,
            "train_test_split": "train",
            "control_type": control,
        }
        for index, control in enumerate(controls)
    ]
    row_packet.write_text(
        json.dumps({"metadata": metadata, "hbsm_entries": primary_entries + control_entries}) + "\n",
        encoding="utf-8",
    )

    build_hbsm_packet(row_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="hbsm")

    assert not report["ok"]
    assert any("top_decile_flag must match measured kl_or_nll_drift" in error for error in report["errors"])


def test_hbsm_checker_rejects_prompt_row_top_decile_flag_mismatch(tmp_path: Path) -> None:
    row_packet = tmp_path / "hbsm_rows.json"
    output_dir = tmp_path / "hbsm_prompt_mismatched_top"
    metadata = _base_trace_metadata("hbsm")
    primary_entries = []
    for prompt_index in range(12):
        for layer in range(20):
            boundary = layer < 10
            primary_entries.append(
                {
                    "prompt_id": f"p{prompt_index}",
                    "layer": layer,
                    "boundary_flag": boundary,
                    "precision_perturbation": "mxfp4_e2m1",
                    "kl_or_nll_drift": float(100 - layer) * 0.01,
                    "cheap_predictor": float(100 - layer),
                    "parameter_count": 1024 + layer,
                    "weight_norm": 0.5 + layer,
                    "top_decile_flag": layer < 2,
                    "random_top_decile": layer in {0, 10},
                    "train_test_split": "train" if layer % 2 == 0 else "test",
                    "control_type": "boundary_only",
                }
            )
    primary_entries[0]["top_decile_flag"] = False
    controls = [
        "perturbation_off",
        "random_flags",
        "layer_index",
        "parameter_count_norm",
        "kl_lens_rank",
        "activation_outlier",
    ]
    control_entries = [
        {
            "prompt_id": f"control_{control}",
            "layer": 100 + index,
            "boundary_flag": False,
            "precision_perturbation": "mxfp4_e2m1",
            "kl_or_nll_drift": 0.0,
            "cheap_predictor": 0.0,
            "parameter_count": 1024,
            "weight_norm": 0.5,
            "top_decile_flag": False,
            "random_top_decile": False,
            "train_test_split": "train",
            "control_type": control,
        }
        for index, control in enumerate(controls)
    ]
    row_packet.write_text(
        json.dumps({"metadata": metadata, "hbsm_entries": primary_entries + control_entries}) + "\n",
        encoding="utf-8",
    )

    build_hbsm_packet(row_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="hbsm")

    assert not report["ok"]
    assert any("every boundary_only prompt row top_decile_flag" in error for error in report["errors"])


def test_hbsm_packet_builder_rejects_string_boolean_flags(tmp_path: Path) -> None:
    row_packet = tmp_path / "hbsm_rows.json"
    output_dir = tmp_path / "hbsm_bad_bool"
    metadata = _base_trace_metadata("hbsm")
    row_packet.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "hbsm_entries": [
                    {
                        "prompt_id": "p0",
                        "layer": 0,
                        "boundary_flag": "false",
                        "precision_perturbation": "mxfp4_e2m1",
                        "kl_or_nll_drift": 0.0,
                        "cheap_predictor": 1.0,
                        "parameter_count": 1024,
                        "weight_norm": 0.5,
                        "top_decile_flag": False,
                        "random_top_decile": False,
                        "train_test_split": "train",
                        "control_type": "perturbation_off",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        build_hbsm_packet(row_packet, output_dir)
    except ValueError as exc:
        assert "boundary_flag must be a boolean" in str(exc)
    else:
        raise AssertionError("expected string boolean flag to be rejected")


def test_gate_packet_checker_rejects_hbsm_without_true_and_false_boundary_flags(tmp_path: Path) -> None:
    row_packet = tmp_path / "hbsm_rows.json"
    output_dir = tmp_path / "hbsm_bad"
    metadata = _base_trace_metadata("hbsm")
    controls = [
        "perturbation_off",
        "random_flags",
        "layer_index",
        "parameter_count_norm",
        "kl_lens_rank",
        "activation_outlier",
    ]
    primary_entries = [
        {
            "prompt_id": f"p{index}",
            "layer": index,
            "boundary_flag": True,
            "precision_perturbation": "mxfp4_e2m1",
            "kl_or_nll_drift": float(index) * 0.01,
            "cheap_predictor": float(index + 1),
            "parameter_count": 1024 + index,
            "weight_norm": 0.5 + index,
            "top_decile_flag": index < 4,
            "random_top_decile": index in {4, 5, 6, 7},
            "train_test_split": "train" if index % 2 == 0 else "test",
            "control_type": "boundary_only",
        }
        for index in range(12)
    ]
    row_packet.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "hbsm_entries": primary_entries + [
                    {
                        "prompt_id": f"control_{control}",
                        "layer": index,
                        "boundary_flag": False,
                        "precision_perturbation": "mxfp4_e2m1",
                        "kl_or_nll_drift": 0.0 if control == "perturbation_off" else float(index) * 0.01,
                        "cheap_predictor": float(index + 1),
                        "parameter_count": 1024 + index,
                        "weight_norm": 0.5 + index,
                        "top_decile_flag": control == "boundary_only",
                        "random_top_decile": control == "random_flags",
                        "train_test_split": "train" if index % 2 == 0 else "test",
                        "control_type": control,
                    }
                    for index, control in enumerate(controls)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    build_hbsm_packet(row_packet, output_dir)
    report = validate_gate_packet(output_dir, mode="real", project="hbsm")

    assert not report["ok"]
    assert any("both boundary_flag=true and boundary_flag=false" in error for error in report["errors"])


def test_hbsm_packet_builder_rejects_top_level_capture_template(tmp_path: Path) -> None:
    row_packet = tmp_path / "hbsm_template.json"
    output_dir = tmp_path / "hbsm_template"
    metadata = _base_trace_metadata("hbsm")
    row_packet.write_text(
        json.dumps(
            {
                "_template_only": True,
                "metadata": metadata,
                "hbsm_entry_templates": [
                    {
                        "prompt_id": "p0",
                        "layer": 0,
                        "boundary_flag": True,
                        "precision_perturbation": "mxfp4_e2m1",
                        "kl_or_nll_drift": "TO_FILL_BEFORE_CAPTURE",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="capture template"):
        build_hbsm_packet(row_packet, output_dir)
