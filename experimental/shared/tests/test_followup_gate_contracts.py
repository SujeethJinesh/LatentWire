import json
from pathlib import Path

from experimental.shared.followup_gate_contracts import (
    EVALUATORS,
    evaluate_hbsm_b2,
    evaluate_hbsm_b3,
    evaluate_horn_h2,
    evaluate_horn_h3,
    evaluate_ssq_lr_s2,
    evaluate_ssq_lr_s3,
    validate_followup_gate_packet,
)


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


def _write_packet(tmp_path: Path, gate: str, rows: list[dict[str, object]]) -> Path:
    output_dir = tmp_path / gate
    output_dir.mkdir(parents=True)
    evaluated = EVALUATORS[gate](rows)
    config = {
        "gate_name": gate,
        "project": gate.split("_")[0],
        "source_gate_packet_sha256": SHA_A,
        "preregistration_sha256": SHA_B,
        "seed_list": [20260506],
        "command": f"pytest synthetic {gate}",
    }
    summary = {
        **evaluated,
        "decision": evaluated["gate_status"],
        "row_count": len(rows),
        "claim_boundary": ["follow-up contract only", "not model evidence"],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "raw_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    (output_dir / "decision.md").write_text(f"# Decision\n\n`{summary['decision']}`\n")
    return output_dir


def _ssq_lr_s2_rows() -> list[dict[str, object]]:
    base = {
        "model_id": "granite-tiny",
        "prompt_id": "p0",
        "recipe_id": "mxfp4_block64",
        "precision": "mxfp4_e2m1",
        "scale_granularity": "per_block",
        "block_size": 64,
        "effective_bits": 4.25,
        "bf16_state_bytes": 16000.0,
        "quantized_state_bytes": 3900.0,
        "scale_bytes": 80.0,
        "metadata_bytes": 20.0,
        "bf16_accuracy": 0.50,
        "quantized_accuracy": 0.495,
        "accuracy_delta_abs": 0.005,
        "bf16_nll": 1.0,
        "quantized_nll": 1.004,
        "nll_delta": 0.004,
        "paired_ci_low": -0.002,
        "paired_ci_high": 0.008,
        "bf16_noop_delta": 0.0,
    }
    rows = [{**base, "prompt_id": "p0", "control_type": "candidate_recipe"}]
    rows.append({**base, "prompt_id": "p1", "control_type": "candidate_recipe"})
    rows.append(
        {
            **base,
            "recipe_id": "bf16_noop",
            "precision": "bf16",
            "quantized_state_bytes": 16000.0,
            "scale_bytes": 0.0,
            "metadata_bytes": 0.0,
            "accuracy_delta_abs": 0.0,
            "nll_delta": 0.0,
            "paired_ci_high": 0.0,
            "control_type": "bf16_noop",
        }
    )
    rows.append(
        {
            **base,
            "recipe_id": "uniform_same_bytes",
            "accuracy_delta_abs": 0.03,
            "paired_ci_high": 0.04,
            "control_type": "same_byte_uniform",
        }
    )
    rows.extend(
        [
            {
                **base,
                "recipe_id": control_type,
                "precision": precision,
                "accuracy_delta_abs": delta,
                "paired_ci_high": delta + 0.005,
                "control_type": control_type,
            }
            for control_type, precision, delta in [
                ("int8_state", "int8_symmetric", 0.004),
                ("fp8_state", "fp8_e4m3", 0.006),
                ("mxfp4_state", "mxfp4_e2m1", 0.005),
                ("random_same_l2", "matched_noise", 0.025),
                ("shuffled_scales", "mxfp4_e2m1_shuffled_scales", 0.020),
            ]
        ]
    )
    return rows


def _ssq_lr_s3_rows() -> list[dict[str, object]]:
    base = {
        "recipe_id": "mxfp4_block64",
        "frozen_recipe_sha256": SHA_A,
        "source_s2_packet_sha256": SHA_B,
        "retuned": False,
        "model_role": "validation",
        "bf16_accuracy": 0.50,
        "quantized_accuracy": 0.485,
        "accuracy_delta_abs": 0.015,
        "paired_ci_high": 0.018,
        "bf16_nll": 1.0,
        "quantized_nll": 1.01,
        "nll_delta": 0.01,
    }
    return [
        {**base, "model_id": "granite", "prompt_id": "p0", "control_type": "transfer_eval"},
        {**base, "model_id": "bamba", "prompt_id": "p0", "control_type": "transfer_eval"},
        {**base, "model_id": "granite", "prompt_id": "p1", "control_type": "retune_probe"},
    ]


def _horn_h2_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in [20260506, 20260507, 20260508]:
        for prompt_id in ["p0", "p1"]:
            for direction, drift in [("ssm->attention", 0.30), ("attention->ssm", 0.10)]:
                rows.append(
                    {
                        "model_id": "granite",
                        "prompt_id": prompt_id,
                        "prompt_cluster_id": prompt_id,
                        "selected_direction_from_h1": "ssm->attention",
                        "boundary_direction": direction,
                        "noise_side": "pre_boundary",
                        "noise_std_basis": "fp4_equivalent",
                        "noise_scale": 1.0,
                        "seed": seed,
                        "clean_nll": 1.0,
                        "noisy_nll": 1.0 + drift,
                        "delta_nll": drift,
                        "hook_off_delta": 0.0,
                        "control_type": "directional_noise",
                    }
                )
    rows.append({**rows[0], "control_type": "hook_off", "hook_off_delta": 0.0})
    rows.append({**rows[1], "control_type": "flipped_direction_label"})
    return rows


def _horn_h3_rows() -> list[dict[str, object]]:
    base = {
        "model_role": "hybrid_validation",
        "architecture_family": "hybrid",
        "prompt_id": "p0",
        "selected_direction_from_h1": "ssm->attention",
        "boundary_direction": "ssm->attention",
        "directional_drift_ratio": 2.0,
        "directional_ratio_ci_low": 1.2,
        "pure_control_expected_null": False,
    }
    return [
        {**base, "model_id": "granite", "control_type": "hybrid_validation"},
        {**base, "model_id": "bamba", "control_type": "hybrid_validation"},
        {
            **base,
            "model_id": "qwen",
            "model_role": "pure_attention_control",
            "architecture_family": "transformer",
            "directional_drift_ratio": 1.1,
            "directional_ratio_ci_low": 0.9,
            "pure_control_expected_null": True,
            "control_type": "pure_attention_control",
        },
        {
            **base,
            "model_id": "mamba",
            "model_role": "pure_mamba_control",
            "architecture_family": "mamba",
            "directional_drift_ratio": 1.0,
            "directional_ratio_ci_low": 0.8,
            "pure_control_expected_null": True,
            "control_type": "pure_mamba_control",
        },
    ]


def _hbsm_b2_rows() -> list[dict[str, object]]:
    selected = {
        "model_id": "granite",
        "predictor_name": "weight_kurtosis_condition",
        "predictor_registry_sha256": SHA_A,
        "hyperparams_sha256": SHA_B,
        "registry_status": "preregistered",
        "selection_split": "train",
        "spearman": 0.66,
        "spearman_ci_low": 0.55,
        "baseline_name": "kl_lens_rank",
        "baseline_spearman": 0.52,
        "baseline_ci_low": 0.40,
        "margin_vs_best_baseline": 0.07,
        "selected_predictor": True,
        "control_type": "candidate_predictor",
    }
    baseline = {
        **selected,
        "predictor_name": "kl_lens_rank",
        "spearman": 0.52,
        "spearman_ci_low": 0.40,
        "margin_vs_best_baseline": 0.0,
        "selected_predictor": False,
        "control_type": "baseline_predictor",
    }
    return [
        {**selected, "train_test_split": "train"},
        {**selected, "train_test_split": "test"},
        {**baseline, "train_test_split": "test"},
    ]


def _hbsm_b3_rows() -> list[dict[str, object]]:
    base = {
        "model_id": "granite",
        "prompt_id": "p0",
        "layer": 4,
        "boundary_direction": "ssm->attention",
        "noise_scale": 1.0,
        "horn_alignment_sign": "attention_gt_ssm",
        "attention_output_drift": 0.30,
        "ssm_output_drift": 0.10,
        "paired_delta": 0.20,
        "paired_ci_low": 0.08,
        "paired_ci_high": 0.30,
    }
    return [
        {**base, "control_type": "matched_noise"},
        {
            **base,
            "attention_output_drift": 0.0,
            "ssm_output_drift": 0.0,
            "paired_delta": 0.0,
            "control_type": "noise_off",
        },
        {**base, "paired_delta": -0.15, "control_type": "direction_flip"},
    ]


def test_followup_contracts_accept_preregistered_fixture_packets_not_project_evidence(
    tmp_path: Path,
) -> None:
    packets = {
        "ssq_lr_s2": (_ssq_lr_s2_rows(), evaluate_ssq_lr_s2),
        "ssq_lr_s3": (_ssq_lr_s3_rows(), evaluate_ssq_lr_s3),
        "horn_h2": (_horn_h2_rows(), evaluate_horn_h2),
        "horn_h3": (_horn_h3_rows(), evaluate_horn_h3),
        "hbsm_b2": (_hbsm_b2_rows(), evaluate_hbsm_b2),
        "hbsm_b3": (_hbsm_b3_rows(), evaluate_hbsm_b3),
    }
    for gate, (rows, evaluator) in packets.items():
        assert evaluator(rows)["gate_pass"] is True
        packet_dir = _write_packet(tmp_path, gate, rows)
        summary = json.loads((packet_dir / "summary.json").read_text(encoding="utf-8"))
        assert "follow-up contract only" in summary["claim_boundary"]
        assert "not model evidence" in summary["claim_boundary"]
        report = validate_followup_gate_packet(packet_dir, gate=gate)
        assert report["ok"], report["errors"]


def test_followup_contract_rejects_missing_required_control(tmp_path: Path) -> None:
    rows = [row for row in _horn_h2_rows() if row["control_type"] != "hook_off"]
    packet_dir = _write_packet(tmp_path, "horn_h2", rows)

    report = validate_followup_gate_packet(packet_dir, gate="horn_h2")

    assert not report["ok"]
    assert any("missing required controls: hook_off" in error for error in report["errors"])


def test_horn_h2_signed_direction_support_blocks_flip_masking() -> None:
    rows = _horn_h2_rows()
    flipped_one_pair = False
    for row in rows:
        if (
            row["control_type"] == "directional_noise"
            and row["prompt_id"] == "p0"
            and row["seed"] == 20260506
        ):
            row["delta_nll"] = 0.05 if row["boundary_direction"] == "ssm->attention" else 0.40
            flipped_one_pair = True
    assert flipped_one_pair

    evaluation = evaluate_horn_h2(rows)

    assert evaluation["selected_direction_support_fraction"] < 1.0
    assert evaluation["gate_pass"] is False


def test_ssq_lr_s3_contract_rejects_retuning(tmp_path: Path) -> None:
    rows = _ssq_lr_s3_rows()
    rows[0]["retuned"] = True
    packet_dir = _write_packet(tmp_path, "ssq_lr_s3", rows)

    report = validate_followup_gate_packet(packet_dir, gate="ssq_lr_s3")

    assert not report["ok"]
    assert any("retuned must be false" in error for error in report["errors"])


def test_hbsm_b2_contract_rejects_predictor_shopping(tmp_path: Path) -> None:
    rows = _hbsm_b2_rows()
    for row in rows:
        if row["selected_predictor"]:
            row["selection_split"] = "test"
    packet_dir = _write_packet(tmp_path, "hbsm_b2", rows)

    report = validate_followup_gate_packet(packet_dir, gate="hbsm_b2")

    assert not report["ok"]
    assert any("selected from train split only" in error for error in report["errors"])


def test_horn_h2_contract_requires_three_seeds_and_direction_pairing(tmp_path: Path) -> None:
    one_seed_rows = [row for row in _horn_h2_rows() if row.get("seed") in {20260506}]
    packet_dir = _write_packet(tmp_path / "one_seed", "horn_h2", one_seed_rows)

    report = validate_followup_gate_packet(packet_dir, gate="horn_h2")

    assert not report["ok"]
    assert any("at least 3 seeds" in error for error in report["errors"])

    unpaired_rows = [
        row
        for row in _horn_h2_rows()
        if not (
            row["control_type"] == "directional_noise"
            and row["seed"] == 20260506
            and row["prompt_id"] == "p0"
            and row["boundary_direction"] == "attention->ssm"
        )
    ]
    packet_dir = _write_packet(tmp_path / "unpaired", "horn_h2", unpaired_rows)

    report = validate_followup_gate_packet(packet_dir, gate="horn_h2")

    assert not report["ok"]
    assert any("pair both boundary directions" in error for error in report["errors"])


def test_horn_h3_contract_rejects_pure_controls_above_preregistered_threshold(tmp_path: Path) -> None:
    rows = _horn_h3_rows()
    for row in rows:
        if row["control_type"] == "pure_attention_control":
            row["directional_drift_ratio"] = 1.3
            row["directional_ratio_ci_low"] = 1.1
    packet_dir = _write_packet(tmp_path, "horn_h3", rows)

    report = validate_followup_gate_packet(packet_dir, gate="horn_h3")

    assert not report["ok"]
    assert any("fold below 1.2" in error for error in report["errors"])


def test_ssq_lr_s2_contract_requires_full_preregistered_baseline_set(tmp_path: Path) -> None:
    rows = [row for row in _ssq_lr_s2_rows() if row["control_type"] != "random_same_l2"]
    packet_dir = _write_packet(tmp_path, "ssq_lr_s2", rows)

    report = validate_followup_gate_packet(packet_dir, gate="ssq_lr_s2")

    assert not report["ok"]
    assert any("random_same_l2" in error for error in report["errors"])


def test_ssq_lr_s2_contract_accepts_preregistered_nll_alternative() -> None:
    rows = _ssq_lr_s2_rows()
    for row in rows:
        if row["control_type"] == "candidate_recipe":
            row["accuracy_delta_abs"] = 0.03
            row["paired_ci_high"] = 0.04
            row["nll_delta"] = 0.004
            row["nll_delta_abs_ci_high"] = 0.008

    result = evaluate_ssq_lr_s2(rows)

    assert result["gate_pass"] is True
    assert result["selected_accuracy_ci_high"] == 0.04
    assert result["selected_nll_delta_ci_high"] == 0.008
