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
from experimental.shared.sensitivity_metrics import kurtosis, rel_l2, spearman_rank_correlation


SSQ_BUCKETS = ("prefill_end", "2k_or_end", "8k_or_end", "final_minus_128")


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
        '"model_id": "toy-hybrid", '
        '"model_revision": "abc123", '
        '"tokenizer_revision": "tok123", '
        '"prompt_source": "fixed_manifest.json", '
        '"prompt_ids_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"seed_list": [1], '
        '"context_lengths": [128], '
        '"dtype": "bf16", '
        '"device": "mps", '
        '"architecture_map_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", '
        '"command": "python run_gate.py"'
        "}\n"
    )
    rows = [
        {
            "model_id": "toy-hybrid",
            "model_revision": "abc123",
            "prompt_id": f"p{prompt_index}",
            "layer": 0,
            "position_bucket": bucket,
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
                    "model_id": "toy-hybrid",
                    "model_revision": "abc123",
                    "prompt_id": f"p{prompt_index}",
                    "layer": 0,
                    "position_bucket": bucket,
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
            "model_id": "toy-hybrid",
            "model_revision": "abc123",
            "prompt_id": f"p{prompt_index}",
            "layer": 0,
            "position_bucket": bucket,
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
            "model_id": "toy-hybrid",
            "model_revision": "abc123",
            "prompt_id": f"p{prompt_index}",
            "layer": 0,
            "position_bucket": "prefill_end",
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
        '"model_id": "toy-hybrid", '
        '"model_revision": "abc123", '
        '"tokenizer_revision": "tok123", '
        '"prompt_source": "fixed_manifest.json", '
        '"prompt_ids_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"seed_list": [1], '
        '"context_lengths": [128], '
        '"dtype": "bf16", '
        '"device": "mps", '
        '"architecture_map_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", '
        '"command": "python run_gate.py"'
        "}\n"
    )
    (packet / "raw_rows.jsonl").write_text(
        "{"
        '"model_id": "toy-hybrid", '
        '"prompt_id": "p0", '
        '"layer_left": 0, '
        '"layer_right": 1, '
        '"direction": "attention->ssm", '
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
    metadata = _base_trace_metadata()
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
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": prompt_id,
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": direction,
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


def _base_trace_metadata() -> dict[str, object]:
    return {
        "model_id": "toy-hybrid",
        "model_revision": "abc123",
        "tokenizer_revision": "tok123",
        "prompt_source": "fixed_manifest.json",
        "prompt_ids_hash": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        "seed_list": [1],
        "context_lengths": [16],
        "dtype": "bf16",
        "device": "cpu",
        "command": "python dump.py",
        "architecture_map_hash": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    }


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
                    "position_bucket": bucket,
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
    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert report["ok"]
    assert report["row_count"] == 48


def test_horn_packet_builder_outputs_required_controls(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn"
    metadata = _base_trace_metadata()
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
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": direction,
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

    assert report["ok"]
    assert report["row_count"] == 72


def test_gate_packet_checker_rejects_horn_permuted_direction_without_matching_boundary(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "horn_bad"
    metadata = _base_trace_metadata()
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
            tensor_name = f"{stem}_p{prompt_index}"
            entries.append(
                {
                    "tensor": tensor_name,
                    "prompt_id": f"p{prompt_index}",
                    "layer_left": layer_left,
                    "layer_right": layer_right,
                    "direction": direction,
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


def test_hbsm_packet_builder_outputs_required_controls(tmp_path: Path) -> None:
    row_packet = tmp_path / "hbsm_rows.json"
    output_dir = tmp_path / "hbsm"
    metadata = _base_trace_metadata()
    primary_entries = []
    for index in range(24):
        boundary = index < 12
        primary_entries.append(
            {
                "prompt_id": f"p{index % 12}",
                "layer": index,
                "boundary_flag": boundary,
                "precision_perturbation": "mxfp4_e2m1",
                "kl_or_nll_drift": float(index) * 0.01,
                "cheap_predictor": float(index + 1),
                "parameter_count": 1024 + index,
                "weight_norm": 0.5 + index,
                "top_decile_flag": boundary and index < 8,
                "random_top_decile": index in {0, 1, 2, 3, 12, 13, 14, 15},
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
    report = validate_gate_packet(output_dir, mode="real", project="hbsm")

    assert report["ok"]
    assert report["row_count"] == 28


def test_hbsm_packet_builder_rejects_string_boolean_flags(tmp_path: Path) -> None:
    row_packet = tmp_path / "hbsm_rows.json"
    output_dir = tmp_path / "hbsm_bad_bool"
    metadata = _base_trace_metadata()
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
    metadata = _base_trace_metadata()
    controls = ["perturbation_off", "random_flags", "layer_index", "parameter_count_norm"]
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
