from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / "experimental/thoughtflow_fp8"


def test_current_decision_manifest_demotes_rdu_topk() -> None:
    manifest = (PROJECT / "phase2/current_decision_manifest_20260506.md").read_text(encoding="utf-8")
    paper = (PROJECT / "paper/thoughtflow_fp8_colm2026.tex").read_text(encoding="utf-8")
    progress = (PROJECT / "progress.md").read_text(encoding="utf-8")

    assert "STOP / diagnostic only" in manifest
    assert "current live method branch: none" in manifest
    assert "no runnable successor gate" in manifest
    assert "R-KV-like was best compressed" in manifest
    assert "psi_topk" in manifest
    assert "vwac_topk" in manifest
    assert "diagnostic, not a positive method" in paper
    assert "LOCAL METHOD EVIDENCE SATURATED / STOP OR PIVOT" in progress


def test_current_decision_manifest_blocks_current_gpu_claims() -> None:
    manifest = (PROJECT / "phase2/current_decision_manifest_20260506.md").read_text(encoding="utf-8")
    reviewer_pack = (PROJECT / "paper/reviewer_pack.md").read_text(encoding="utf-8")

    for text in (manifest, reviewer_pack):
        assert "not a real FP8 method" in text or "no real FP8 serving result exists" in text
        assert "spending GPU time on the current branch" in text or "Stop local method experimentation" in text


def test_reviewer_pack_covers_colm_review_axes_and_no_active_successor() -> None:
    reviewer_pack = (PROJECT / "paper/reviewer_pack.md").read_text(encoding="utf-8")

    for axis in (
        "Benchmarks",
        "Ablations",
        "Correctness",
        "Reproducibility",
        "Novelty",
        "Camera-readiness",
    ):
        assert f"| {axis} |" in reviewer_pack

    assert "No fresh utility signal is currently pre-registered" in reviewer_pack
    assert "psi_topk" in reviewer_pack
    assert "vwac_topk" in reviewer_pack
    assert "Until that artifact exists, there is no runnable successor gate" in reviewer_pack
