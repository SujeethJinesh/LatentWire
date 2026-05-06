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
from experimental.shared.sensitivity_metrics import kurtosis, rel_l2, spearman_rank_correlation


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
        '"prompt_ids_hash": "sha256:abc", '
        '"seed_list": [1], '
        '"context_lengths": [128], '
        '"dtype": "bf16", '
        '"device": "mps", '
        '"architecture_map_hash": "sha256:map", '
        '"command": "python run_gate.py"'
        "}\n"
    )
    row = (
        "{"
        '"model_id": "toy-hybrid", '
        '"model_revision": "abc123", '
        '"prompt_id": "p0", '
        '"layer": 0, '
        '"position_bucket": "early", '
        '"state_shape": [1, 4], '
        '"max_abs": 2.0, '
        '"rms": 1.0, '
        '"std": 0.5, '
        '"kurtosis": 3.0, '
        '"outlier_mass": 0.1, '
        '"control_type": "bf16_no_quant"'
        "}\n"
    )
    (packet / "raw_rows.jsonl").write_text(row)
    (packet / "summary.json").write_text(
        "{"
        '"seed": 1, '
        '"surface": "real_ssq_lr_s1", '
        '"decision": "CONTINUE_REAL_STATE_DUMPS", '
        '"row_count": 1, '
        '"rows": [{"layer": 0}], '
        '"claim_boundary": ["real model trace", "not GPU evidence"]'
        "}\n"
    )
    (packet / "summary.md").write_text("# Summary\n")
    (packet / "decision.md").write_text("`CONTINUE_REAL_STATE_DUMPS`\n")

    report = validate_gate_packet(packet, mode="real", project="ssq_lr")

    assert report["ok"]
    assert report["mode"] == "real"


def test_gate_packet_checker_rejects_real_packet_without_project_controls(tmp_path: Path) -> None:
    packet = tmp_path / "real_horn"
    packet.mkdir()
    (packet / "config.json").write_text(
        "{"
        '"model_id": "toy-hybrid", '
        '"model_revision": "abc123", '
        '"tokenizer_revision": "tok123", '
        '"prompt_source": "fixed_manifest.json", '
        '"prompt_ids_hash": "sha256:abc", '
        '"seed_list": [1], '
        '"context_lengths": [128], '
        '"dtype": "bf16", '
        '"device": "mps", '
        '"architecture_map_hash": "sha256:map", '
        '"command": "python run_gate.py"'
        "}\n"
    )
    (packet / "raw_rows.jsonl").write_text(
        "{"
        '"model_id": "toy-hybrid", '
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
