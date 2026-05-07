import json
from pathlib import Path

from experimental.hbsm.phase2.hbsm_synthetic_b1_gate import run_gate as run_hbsm
from experimental.horn.phase2.horn_synthetic_h1_gate import run_gate as run_horn
from experimental.ssq_lr.phase2.ssq_lr_synthetic_s1_gate import run_gate as run_ssq_lr


def _assert_packet(output_dir: Path, expected_decision: str) -> dict:
    for name in ["config.json", "raw_rows.jsonl", "summary.json", "decision.md"]:
        assert (output_dir / name).exists()
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["decision"] == expected_decision
    assert "synthetic-only" in summary["claim_boundary"]
    return summary


def test_ssq_lr_synthetic_s1_packet(tmp_path: Path) -> None:
    summary = run_ssq_lr(output_dir=tmp_path)
    on_disk = _assert_packet(tmp_path, "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_SSQ_LR_S1")

    assert summary["max_abs_ratio_final_minus_128_vs_prefill_end"] >= 2.0
    assert on_disk["std_ratio_final_minus_128_vs_prefill_end"] >= 2.0
    assert on_disk["distribution_effect_floor_pass"] is True


def test_horn_synthetic_h1_packet(tmp_path: Path) -> None:
    summary = run_horn(output_dir=tmp_path)
    on_disk = _assert_packet(tmp_path, "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HORN_H1A")

    assert summary["max_abs_direction_ratio"] >= 3.0
    assert on_disk["kurtosis_direction_ratio"] >= 1.5
    assert on_disk["selected_h1_cluster_bootstrap_low"] > 0.0


def test_hbsm_synthetic_b1_packet(tmp_path: Path) -> None:
    summary = run_hbsm(output_dir=tmp_path)
    on_disk = _assert_packet(tmp_path, "SCHEMA_REHEARSAL_NOT_PROMOTABLE_SYNTHETIC_HBSM_B1")

    assert summary["cheap_predictor_spearman"] >= 0.6
    assert on_disk["boundary_top_decile_count"] >= 1
