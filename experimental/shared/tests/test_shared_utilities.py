import hashlib
import json
from pathlib import Path

import torch
import pytest

from experimental.shared.activation_dumper import load_tensor_manifest, load_tensor_packet, save_tensor_packet
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
from experimental.shared.hbsm_local_sensitivity_runner import (
    _random_top_layers,
    _replace_first_tensor,
    _top_decile_layers,
    select_hbsm_entries,
)
from experimental.shared.hybrid_manifest_local_capture_runner import (
    _bucket_tokenized,
    _filled_metadata,
    _first_tensor,
    _horn_tensors,
    _install_horn_right_input_hooks,
    select_horn_entries,
    select_ssq_entries,
)
from experimental.shared.hybrid_model_eligibility import (
    _architecture_hash,
    _local_cache_dir,
    _mac_trace_decision,
    _size_gb,
)
import experimental.shared.hybrid_local_capture_preflight as local_capture_preflight
from experimental.shared.hybrid_transformers_smoke_probe import summarize_cache
from experimental.shared.hybrid_trace_packet_builder import build_hbsm_packet, build_horn_packet, build_ssq_lr_packet
from experimental.shared.hybrid_trace_capture_manifest import build_capture_manifests
from experimental.shared.hybrid_trace_plan import write_trace_plan
from experimental.shared.sensitivity_metrics import kurtosis, rel_l2, spearman_rank_correlation


SSQ_BUCKETS = ("prefill_end", "2k_or_end", "8k_or_end", "final_minus_128")
TRACE_PLAN_HASHES = {
    "ssq_lr": "sha256:a05dab6ad3b821b91bd2e3c67340703bd7c7594e8d86b79051bfe763da17305b",
    "horn": "sha256:a2df7d6485d376747ba179c80172882b3dddd440d1db3b5f765f777a857e75f0",
    "hbsm": "sha256:015e28d426aa4c11d00c67234c15e5cf5ed8f599a28de102fbc00aaccc84ed67",
}
GRANITE_TINY_REVISION = "791e0d3d28c86e106c9b6e0b4cecdee0375b6124"
HBSM_CONTROLS = (
    "perturbation_off",
    "random_flags",
    "layer_index",
    "parameter_count_norm",
    "kl_lens_rank",
    "activation_outlier",
)


def _hbsm_layer_aligned_controls(primary_entries: list[dict[str, object]]) -> list[dict[str, object]]:
    by_layer: dict[int, dict[str, object]] = {}
    for entry in primary_entries:
        layer = int(entry["layer"])
        by_layer.setdefault(layer, entry)
    controls: list[dict[str, object]] = []
    for control in HBSM_CONTROLS:
        for layer, primary in sorted(by_layer.items()):
            controls.append(
                {
                    "prompt_id": f"control_{control}",
                    "layer": layer,
                    "boundary_flag": bool(primary["boundary_flag"]),
                    "precision_perturbation": "perturbation_off"
                    if control == "perturbation_off"
                    else "mxfp4_e2m1",
                    "kl_or_nll_drift": 0.0
                    if control == "perturbation_off"
                    else float(primary["kl_or_nll_drift"]),
                    "cheap_predictor": float(primary.get("cheap_predictor", 0.0)),
                    "parameter_count": int(primary.get("parameter_count", 1024)),
                    "weight_norm": float(primary.get("weight_norm", 0.5)),
                    "top_decile_flag": False,
                    "random_top_decile": False,
                    "train_test_split": "control",
                    "control_type": control,
                }
            )
    return controls


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


def test_tensor_packet_manifest_records_original_name_and_sha256(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    save_tensor_packet(
        packet,
        tensors={"layer/0 state": torch.arange(6).reshape(2, 3)},
        metadata={"model": "toy", "trace_count": 1},
    )

    manifest = load_tensor_manifest(packet)
    row = manifest["layer_0_state"]
    expected_hash = "sha256:" + hashlib.sha256((packet / "layer_0_state.pt").read_bytes()).hexdigest()

    assert row["original_name"] == "layer/0 state"
    assert row["storage_name"] == "layer_0_state.pt"
    assert row["shape"] == [2, 3]
    assert row["sha256"] == expected_hash


def test_tensor_packet_rejects_sanitized_name_collisions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="collide"):
        save_tensor_packet(
            tmp_path / "packet",
            tensors={"layer/0": torch.ones(1), "layer_0": torch.zeros(1)},
            metadata={"model": "toy"},
        )


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

    assert report["ok"], report["errors"]
    assert report["row_count"] == 1


def test_gate_packet_checker_rejects_missing_files(tmp_path: Path) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()

    report = validate_gate_packet(packet)

    assert not report["ok"]
    assert "missing config.json" in report["errors"]


def test_local_capture_preflight_reports_ready_when_cache_and_deps_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    architecture_maps = tmp_path / "architecture_maps.json"
    capture_summary = tmp_path / "capture_summary.json"
    eligibility_summary = tmp_path / "eligibility_summary.json"
    cache_root = tmp_path / "hf"
    snapshot = cache_root / "hub/models--org--hybrid/snapshots/rev"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    architecture_maps.write_text(
        json.dumps(
            [
                {
                    "model_id": "org-hybrid",
                    "model_id_aliases": ["org/hybrid", "org-hybrid"],
                    "model_type": "granitemoehybrid",
                    "architecture": "ToyHybrid",
                    "config_sha256": "a" * 64,
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "boundary_count": 1,
                    "direction_counts": {"ssm->attention": 1},
                }
            ]
        )
    )
    capture_summary.write_text(json.dumps({"counts": {"ssq_lr": {"org-hybrid": 4}}}))
    eligibility_summary.write_text(json.dumps({"rows": [{"model_id": "org/hybrid", "safetensors_gb": 0.001}]}))
    monkeypatch.setattr(
        local_capture_preflight,
        "_package_status",
        lambda package_names: {name: {"installed": True, "origin": "test"} for name in package_names},
    )
    monkeypatch.setattr(
        local_capture_preflight,
        "_transformers_model_class_status",
        lambda aliases: (True, "toy:ToyHybridForCausalLM"),
    )

    summary = local_capture_preflight.write_packet(
        output_dir=tmp_path / "out",
        architecture_maps=architecture_maps,
        capture_summary=capture_summary,
        eligibility_summary=eligibility_summary,
        cache_roots=(cache_root,),
        mac_weight_budget_gb=1.0,
    )

    assert summary["decision"] == "LOCAL_CAPTURE_READY_NOT_EVIDENCE"
    assert summary["rows"][0]["weights_cached"] is True
    assert "not model evidence" in summary["claim_boundary"]


def test_local_capture_preflight_does_not_block_on_optional_mamba_ssm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    architecture_maps = tmp_path / "architecture_maps.json"
    capture_summary = tmp_path / "capture_summary.json"
    eligibility_summary = tmp_path / "eligibility_summary.json"
    cache_root = tmp_path / "hf"
    snapshot = cache_root / "hub/models--org--hybrid/snapshots/rev"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    architecture_maps.write_text(
        json.dumps(
            [
                {
                    "model_id": "org-hybrid",
                    "model_id_aliases": ["org/hybrid"],
                    "model_type": "granitemoehybrid",
                    "architecture": "ToyHybrid",
                    "config_sha256": "a" * 64,
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "boundary_count": 1,
                    "direction_counts": {"ssm->attention": 1},
                }
            ]
        )
    )
    capture_summary.write_text(json.dumps({"counts": {"horn": {"org-hybrid": 2}}}))
    eligibility_summary.write_text(json.dumps({"rows": [{"model_id": "org/hybrid", "safetensors_gb": 0.001}]}))

    def fake_package_status(package_names: tuple[str, ...]) -> dict[str, dict[str, object]]:
        return {
            name: {"installed": name != "mamba_ssm", "origin": "test" if name != "mamba_ssm" else None}
            for name in package_names
        }

    monkeypatch.setattr(local_capture_preflight, "_package_status", fake_package_status)
    monkeypatch.setattr(
        local_capture_preflight,
        "_transformers_model_class_status",
        lambda aliases: (True, "toy:ToyHybridForCausalLM"),
    )

    summary = local_capture_preflight.write_packet(
        output_dir=tmp_path / "out",
        architecture_maps=architecture_maps,
        capture_summary=capture_summary,
        eligibility_summary=eligibility_summary,
        cache_roots=(cache_root,),
        mac_weight_budget_gb=1.0,
    )

    assert summary["decision"] == "LOCAL_CAPTURE_READY_NOT_EVIDENCE"
    assert summary["rows"][0]["optional_runtime_packages_missing"] == ["mamba_ssm"]
    assert summary["rows"][0]["blocking_reasons"] == []


def test_local_capture_preflight_blocks_when_transformers_class_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    architecture_maps = tmp_path / "architecture_maps.json"
    capture_summary = tmp_path / "capture_summary.json"
    eligibility_summary = tmp_path / "eligibility_summary.json"
    cache_root = tmp_path / "hf"
    snapshot = cache_root / "hub/models--org--hybrid/snapshots/rev"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    architecture_maps.write_text(
        json.dumps(
            [
                {
                    "model_id": "org-hybrid",
                    "model_id_aliases": ["org/hybrid"],
                    "model_type": "granitemoehybrid",
                    "architecture": "ToyHybrid",
                    "config_sha256": "a" * 64,
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "boundary_count": 1,
                    "direction_counts": {"ssm->attention": 1},
                }
            ]
        )
    )
    capture_summary.write_text(json.dumps({"counts": {"hbsm": {"org-hybrid": 2}}}))
    eligibility_summary.write_text(json.dumps({"rows": [{"model_id": "org/hybrid", "safetensors_gb": 0.001}]}))
    monkeypatch.setattr(
        local_capture_preflight,
        "_package_status",
        lambda package_names: {name: {"installed": True, "origin": "test"} for name in package_names},
    )
    monkeypatch.setattr(
        local_capture_preflight,
        "_transformers_model_class_status",
        lambda aliases: (False, "toy class missing"),
    )

    summary = local_capture_preflight.write_packet(
        output_dir=tmp_path / "out",
        architecture_maps=architecture_maps,
        capture_summary=capture_summary,
        eligibility_summary=eligibility_summary,
        cache_roots=(cache_root,),
        mac_weight_budget_gb=1.0,
    )

    assert summary["decision"] == "LOCAL_CAPTURE_BLOCKED_DEPS_NOT_EVIDENCE"
    assert "local transformers cannot instantiate" in summary["rows"][0]["blocking_reasons"][0]


def test_transformers_smoke_probe_summarizes_recurrent_and_attention_cache() -> None:
    class Layer:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Cache:
        layers = [
            Layer(recurrent_states=torch.zeros(1, 2, 3), conv_states=torch.zeros(1, 4, 5)),
            Layer(keys=torch.zeros(1, 2, 3, 4), values=torch.zeros(1, 2, 3, 4)),
        ]

    summary = summarize_cache(Cache())

    assert summary["cache_layer_count"] == 2
    assert summary["recurrent_state_layer_count"] == 1
    assert summary["attention_cache_layer_count"] == 1
    assert summary["sampled_layers"][0]["fields"]["recurrent_states"] == [1, 2, 3]


def test_manifest_local_capture_selects_minimum_safe_resource_limited_entries() -> None:
    ssq_template = {
        "model_id": "ibm-granite-4.0-h-tiny",
        "trace_plan_hash": TRACE_PLAN_HASHES["ssq_lr"],
        "ssq_lr_entries": [
            {
                "tensor": f"ssq/p0/layer_0/{bucket}",
                "prompt_id": "p0",
                "layer": 0,
                "layer_kind": "ssm",
                "position_bucket": bucket,
                "state_tensor_kind": "mamba2_recurrent_state",
                "control_type": "bf16_no_quant",
            }
            for bucket in SSQ_BUCKETS
        ],
    }
    assert [entry["position_bucket"] for entry in select_ssq_entries(ssq_template, prompt_id="p0")] == list(
        SSQ_BUCKETS
    )

    horn_template = {
        "horn_entries": [
            {
                "tensor": f"horn/p0/boundary_{boundary}/{control}",
                "prompt_id": "p0",
                "prompt_cluster_id": "cluster",
                "layer_left": boundary,
                "layer_right": boundary + 1,
                "direction": direction if control != "permuted_direction" else flipped,
                "matched_boundary_direction": direction if control != "permuted_direction" else flipped,
                "boundary_index": boundary,
                "pre_norm_position": "post",
                "post_norm_position": "pre",
                "control_type": control,
                **({"tensor_alias_of": f"horn/p0/boundary_{boundary}/boundary"} if control == "permuted_direction" else {}),
            }
            for boundary, direction, flipped in (
                (0, "ssm->attention", "attention->ssm"),
                (1, "attention->ssm", "ssm->attention"),
            )
            for control in ("boundary", "non_boundary", "permuted_direction")
        ]
    }
    selected = select_horn_entries(horn_template, prompt_id="p0")
    assert len(selected) == 6
    assert {entry["control_type"] for entry in selected} == {
        "boundary",
        "non_boundary",
        "permuted_direction",
    }

    metadata = _filled_metadata(
        ssq_template,
        project="ssq_lr",
        entries=ssq_template["ssq_lr_entries"],
        max_input_tokens=8,
        prompt_count=1,
    )
    assert metadata["resource_limit_note"].startswith("ssq_lr local runner used 1 prompt")
    assert metadata["model_revision"] == GRANITE_TINY_REVISION
    assert "_template_only" not in metadata


def test_manifest_local_capture_horn_hooks_capture_right_layer_inputs() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [
                    torch.nn.Identity(),
                    torch.nn.Identity(),
                    torch.nn.Identity(),
                ]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.model.layers:
                x = layer(x + 1.0)
            return x

    entries = [
        {
            "tensor": "horn/p0/boundary_0/observed",
            "control_type": "boundary",
            "layer_right": 1,
        },
        {
            "tensor": "horn/p0/boundary_0/permuted_label",
            "tensor_alias_of": "horn/p0/boundary_0/observed",
            "control_type": "permuted_direction",
            "layer_right": 1,
        },
        {
            "tensor": "horn/p0/non_boundary_0_1/boundary_0",
            "control_type": "non_boundary",
            "layer_right": 2,
        },
    ]
    model = ToyModel()
    captures, handles = _install_horn_right_input_hooks(model, entries)
    try:
        _ = model(torch.zeros(1, 2))
    finally:
        for handle in handles:
            handle.remove()

    tensors = _horn_tensors(captures, entries)

    assert torch.equal(tensors["horn/p0/boundary_0/observed"], torch.full((1, 2), 2.0))
    assert torch.equal(tensors["horn/p0/non_boundary_0_1/boundary_0"], torch.full((1, 2), 3.0))
    assert "horn/p0/boundary_0/permuted_label" not in tensors


def test_manifest_local_capture_first_tensor_handles_nested_inputs() -> None:
    expected = torch.ones(2)
    assert _first_tensor(({"mask": None, "hidden": [expected]},)) is expected


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
    for index, row in enumerate(rows):
        row.update(_fake_tensor_provenance(f"state_{index}", row["state_shape"]))
    _write_fake_tensor_artifacts(packet, rows)
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

    assert report["ok"], report["errors"]
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
    for index, row in enumerate(rows):
        row.update(_fake_tensor_provenance(f"state_{index}", row["state_shape"]))
    _write_fake_tensor_artifacts(packet, rows)
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
    _attach_trace_plan_path_from_raw_rows(packet, tmp_path, "ssq_lr")

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
    for index, row in enumerate(rows):
        row.update(_fake_tensor_provenance(f"state_{index}", row["state_shape"]))
    _write_fake_tensor_artifacts(packet, rows)
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
    _attach_trace_plan_path_from_raw_rows(packet, tmp_path, "ssq_lr")

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
    assert summary["row_counts"] == {"hbsm": 32, "horn": 12, "ssq_lr": 24}
    horn_rows = [
        json.loads(line)
        for line in (output_dir / "horn_trace_plan.jsonl").read_text().splitlines()
    ]
    assert {row["control_type"] for row in horn_rows} == {
        "boundary",
        "non_boundary",
        "permuted_direction",
    }
    assert {row["prompt_cluster_id"] for row in horn_rows} == {"p0", "p1"}
    assert {
        row["matched_boundary_direction"]
        for row in horn_rows
        if row["control_type"] == "non_boundary"
    } == {"attention->ssm", "ssm->attention"}
    for row in horn_rows:
        if row["control_type"] == "permuted_direction":
            observed_name = row["tensor_name"].replace("/permuted_label", "/observed")
            assert row["tensor_alias_of"] == observed_name
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
    assert {entry["prompt_cluster_id"] for entry in horn_template["horn_entries"]} == {"p0", "p1"}
    assert "hbsm_entries" not in hbsm_template
    assert len(hbsm_template["hbsm_entry_templates"]) == 32
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
    assert (
        _mac_trace_decision(
            weights_cached=True,
            estimated_weight_gb=8.0,
            requires_mamba_ssm=True,
            mamba_ssm_installed=False,
        )
        == "POSSIBLE_LOCAL_CACHE_CHECK_REQUIRED"
    )


def test_hbsm_select_entries_keeps_layer_aligned_controls() -> None:
    entries = []
    for layer in range(4):
        boundary = layer in {1, 2}
        entries.append(
            {
                "prompt_id": "p0",
                "layer": layer,
                "boundary_flag": boundary,
                "control_type": "boundary_only",
            }
        )
    for control_type in HBSM_CONTROLS:
        for layer in range(4):
            entries.append(
                {
                    "prompt_id": f"control_{control_type}",
                    "layer": layer,
                    "boundary_flag": layer in {1, 2},
                    "control_type": control_type,
                }
            )

    selected = select_hbsm_entries(
        {"hbsm_entry_templates": entries},
        prompt_id="p0",
        layer_limit=4,
    )

    assert len(selected) == 4 * (1 + len(HBSM_CONTROLS))
    assert {
        row["control_type"]
        for row in selected
    } == {"boundary_only", *HBSM_CONTROLS}


def test_hbsm_replace_first_tensor_preserves_nested_structure() -> None:
    value = ("meta", {"later": torch.ones(2), "first": torch.arange(2, dtype=torch.float32)})

    replaced, did_replace = _replace_first_tensor(value, lambda tensor: tensor + 10)

    assert did_replace
    assert isinstance(replaced, tuple)
    assert torch.equal(replaced[1]["first"], torch.tensor([10.0, 11.0]))
    assert torch.equal(replaced[1]["later"], torch.ones(2))


def test_hbsm_top_and_random_decile_are_same_cardinality() -> None:
    drift = {layer: float(layer) for layer in range(11)}
    primary_by_layer = {
        layer: {"boundary_flag": layer % 2 == 0}
        for layer in drift
    }

    top_layers = _top_decile_layers(drift)
    random_layers = _random_top_layers(primary_by_layer, top_layers)

    assert top_layers == {10, 9}
    assert len(random_layers) == len(top_layers)
    assert not (top_layers & random_layers)


def test_ssq_bucket_tokenized_uses_monotone_short_prefixes() -> None:
    tokenized = {
        "input_ids": torch.arange(8).reshape(1, 8),
        "attention_mask": torch.ones(1, 8),
    }

    lengths = [
        int(_bucket_tokenized(tokenized, bucket=bucket, max_input_tokens=8)["input_ids"].shape[1])
        for bucket in SSQ_BUCKETS
    ]

    assert lengths == [2, 4, 6, 8]


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
    config["resource_limit_note"] = "unit-test packet covers a deliberately small trace-plan subset"
    (packet_dir / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    summary = json.loads((packet_dir / "summary.json").read_text(encoding="utf-8"))
    gate_status = str(summary.get("gate_status", ""))
    if gate_status:
        summary["decision"] = f"RESOURCE_LIMITED_NOT_PROMOTABLE_{gate_status}"
        (packet_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
        (packet_dir / "decision.md").write_text(summary["decision"] + "\n", encoding="utf-8")


def _fake_tensor_provenance(name: str, shape: list[int] | None = None) -> dict[str, object]:
    tensor_shape = shape or [1, 4]
    safe_name = name.replace("/", "_").replace(" ", "_")
    return {
        "tensor_name": name,
        "tensor_source_name": name,
        "tensor_storage_name": f"{safe_name}.pt",
        "tensor_sha256": "sha256:" + ("a" * 64),
        "tensor_dtype": "torch.float32",
        "tensor_shape": tensor_shape,
    }


def _write_fake_tensor_artifacts(packet_dir: Path, rows: list[dict[str, object]]) -> None:
    tensor_dir = packet_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for index, row in enumerate(rows):
        storage_name = str(row["tensor_storage_name"])
        shape = list(row["tensor_shape"])
        tensor = torch.arange(1, int(torch.prod(torch.tensor(shape))) + 1, dtype=torch.float32).reshape(shape)
        tensor = tensor + float(index)
        tensor_path = tensor_dir / storage_name
        torch.save(tensor, tensor_path)
        sha256 = "sha256:" + hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        row["tensor_sha256"] = sha256
        row["tensor_shape"] = shape
        if "state_shape" in row:
            row["state_shape"] = shape
        values = tensor.float()
        row["max_abs"] = float(torch.amax(torch.abs(values)))
        row["rms"] = float(torch.sqrt(torch.mean(values * values)))
        row["std"] = float(torch.std(values, unbiased=False))
        row["kurtosis"] = kurtosis(values)
        if "outlier_mass" in row:
            flat = values.reshape(-1).abs()
            threshold = flat.mean() + 3.0 * flat.std(unbiased=False)
            row["outlier_mass"] = float(torch.mean((flat > threshold).float()))
        manifest[Path(storage_name).stem] = {
            "dtype": row["tensor_dtype"],
            "original_name": row["tensor_source_name"],
            "sha256": sha256,
            "shape": row["tensor_shape"],
            "storage_name": storage_name,
        }
    (tensor_dir / "tensor_manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (tensor_dir / "metadata.json").write_text("{}\n")


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
    first_row = json.loads((output_dir / "raw_rows.jsonl").read_text().splitlines()[0])

    assert report["ok"], report["errors"]
    assert report["row_count"] == 48
    assert first_row["tensor_name"] == "state_layer_0_prefill_end_p0"
    assert first_row["tensor_source_name"] == "state_layer_0_prefill_end_p0"
    assert first_row["tensor_sha256"].startswith("sha256:")
    assert first_row["tensor_shape"] == [2, 4]
    assert (output_dir / "tensors/tensor_manifest.json").is_file()
    assert (output_dir / "tensors" / first_row["tensor_storage_name"]).is_file()


def test_gate_packet_checker_rejects_missing_saved_tensor_artifact(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_missing_tensor_artifact"
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
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "ssq_lr")
    first_row = json.loads((output_dir / "raw_rows.jsonl").read_text().splitlines()[0])
    (output_dir / "tensors" / first_row["tensor_storage_name"]).unlink()

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("tensor artifact is missing" in error for error in report["errors"])


def test_gate_packet_checker_rejects_saved_tensor_hash_mismatch(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_tensor_hash_mismatch"
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
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "ssq_lr")
    first_row = json.loads((output_dir / "raw_rows.jsonl").read_text().splitlines()[0])
    (output_dir / "tensors" / first_row["tensor_storage_name"]).write_bytes(b"corrupted tensor bytes")

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("tensor_sha256 does not match" in error for error in report["errors"])


def test_gate_packet_checker_rejects_metric_that_disagrees_with_saved_tensor(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_metric_mismatch"
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
    _attach_trace_plan_path_from_raw_rows(output_dir, tmp_path, "ssq_lr")
    rows = [json.loads(line) for line in (output_dir / "raw_rows.jsonl").read_text().splitlines()]
    rows[0]["max_abs"] = float(rows[0]["max_abs"]) + 123.0
    (output_dir / "raw_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("max_abs does not match loaded tensor" in error for error in report["errors"])


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
    assert report["ok"], report["errors"]


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


def test_gate_packet_checker_rejects_promotable_packet_with_unpinned_trace_plan_path(tmp_path: Path) -> None:
    tensor_packet = tmp_path / "tensor_packet"
    output_dir = tmp_path / "ssq_unpinned_trace_plan_path"
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
    trace_plan = tmp_path / "caller_supplied_subset_trace_plan.jsonl"
    trace_plan.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    config["trace_plan_path"] = str(trace_plan)
    (output_dir / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")

    report = validate_gate_packet(output_dir, mode="real", project="ssq_lr")

    assert not report["ok"]
    assert any("trace_plan_path must point to rows whose SHA-256" in error for error in report["errors"])


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

    assert report["ok"], report["errors"]
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

    assert report["ok"], report["errors"]
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
    config["resource_limit_note"] = "unit-test packet covers a deliberately small trace-plan subset"
    (output_dir / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    summary["decision"] = f"RESOURCE_LIMITED_NOT_PROMOTABLE_{summary['gate_status']}"
    (output_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True) + "\n")
    (output_dir / "decision.md").write_text(summary["decision"] + "\n", encoding="utf-8")

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
            entries.append(entry)
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
    rows = [json.loads(line) for line in (output_dir / "raw_rows.jsonl").read_text().splitlines()]
    boundary = next(
        row
        for row in rows
        if row["control_type"] == "boundary" and row["boundary_index"] == 0 and row["prompt_id"] == "p0"
    )
    permuted = next(
        row
        for row in rows
        if row["control_type"] == "permuted_direction"
        and row["boundary_index"] == 0
        and row["prompt_id"] == "p0"
    )

    assert report["ok"], report["errors"]
    assert report["row_count"] == 72
    assert permuted["tensor_alias_of"] == boundary["tensor_name"]
    assert permuted["tensor_source_name"] == boundary["tensor_source_name"]
    assert permuted["tensor_sha256"] == boundary["tensor_sha256"]
    assert (output_dir / "tensors/tensor_manifest.json").is_file()
    assert (output_dir / "tensors" / boundary["tensor_storage_name"]).is_file()


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
    control_entries = _hbsm_layer_aligned_controls(primary_entries)
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
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))

    assert report["ok"], report["errors"]
    assert report["row_count"] == 420
    assert config["source_row_packet_sha256"].startswith("sha256:")
    assert (output_dir / "evidence/hbsm_row_packet.json").is_file()
    assert (output_dir / "evidence/source_manifest.json").is_file()
    (output_dir / "evidence/hbsm_row_packet.json").unlink()
    missing_report = validate_gate_packet(output_dir, mode="real", project="hbsm")
    assert not missing_report["ok"]
    assert any("evidence/hbsm_row_packet.json" in error for error in missing_report["errors"])


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
    control_entries = _hbsm_layer_aligned_controls(primary_entries)
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
    control_entries = _hbsm_layer_aligned_controls(primary_entries)
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
                "hbsm_entries": primary_entries + _hbsm_layer_aligned_controls(primary_entries),
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
