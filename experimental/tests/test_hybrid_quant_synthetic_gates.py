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
    on_disk = _assert_packet(tmp_path, "SYNTHETIC_PASS_REAL_STATE_DUMPS_NEXT")

    assert summary["max_abs_ratio_late_vs_early"] >= 2.0
    assert on_disk["std_ratio_late_vs_early"] >= 2.0


def test_horn_synthetic_h1_packet(tmp_path: Path) -> None:
    summary = run_horn(output_dir=tmp_path)
    on_disk = _assert_packet(tmp_path, "SYNTHETIC_PASS_REAL_BOUNDARY_DUMPS_NEXT")

    assert summary["ssm_to_attention_over_attention_to_ssm_max_ratio"] >= 3.0
    assert on_disk["ssm_to_attention_over_attention_to_ssm_kurtosis_ratio"] >= 2.0


def test_hbsm_synthetic_b1_packet(tmp_path: Path) -> None:
    summary = run_hbsm(output_dir=tmp_path)
    on_disk = _assert_packet(tmp_path, "SYNTHETIC_PASS_REAL_LAYER_SENSITIVITY_NEXT")

    assert summary["spearman_rho_kurtosis_vs_sensitivity"] >= 0.6
    assert on_disk["boundary_top_decile_hits"] >= 1
