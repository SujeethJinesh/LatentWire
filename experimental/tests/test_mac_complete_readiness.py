from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL = ROOT / "experimental"


def test_top_level_readiness_index_records_current_stop_conditions() -> None:
    readme = (EXPERIMENTAL / "README.md").read_text()

    for phrase in [
        "Experimental Project Control Plane",
        "HybridKernel",
        "SSQ-LR",
        "HORN",
        "HBSM",
        "ThoughtFlow-FP8",
        "These utilities support Mac-local hypothesis gates",
    ]:
        assert phrase in readme


def test_historical_mac_complete_audit_is_marked_superseded() -> None:
    audit = (EXPERIMENTAL / "mac_complete_readiness_20260506.md").read_text()

    assert "Superseded on 2026-05-07" in audit
    assert "HybridKernel, SSQ-LR, HORN, HBSM, and ThoughtFlow-FP8" in audit
    assert "active status authority" in audit


def test_native_handoff_map_has_only_active_project_gates() -> None:
    handoff = (EXPERIMENTAL / "native_gpu_handoff_20260506.md").read_text()

    for phrase in [
        "HybridKernel profiler packet",
        "SSQ-LR",
        "HORN",
        "HBSM",
        "ThoughtFlow-FP8",
        "--require-full-matrix",
        "--require-token-counts",
        "no GPU work for the current branch set",
        "recoverable-gain upper bound clears",
    ]:
        assert phrase in handoff
    for phrase in ["SinkKV", "SinkAware rank-2 native timing"]:
        assert phrase not in handoff


def test_colm_style_pdfs_are_present_and_bounded_for_active_projects() -> None:
    expected_pages = {
        "hybridkernel/paper/hybridkernel_colm2026.pdf": 4,
        "ssq_lr/paper/ssq_lr_colm2026.pdf": 2,
        "horn/paper/horn_colm2026.pdf": 3,
        "hbsm/paper/hbsm_colm2026.pdf": 3,
        "thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf": 8,
    }
    for relative_path, page_count in expected_pages.items():
        pdf_path = EXPERIMENTAL / relative_path
        assert pdf_path.exists()
        assert len(PdfReader(str(pdf_path)).pages) == page_count


def test_project_papers_keep_required_claim_boundaries() -> None:
    required_boundaries = {
        "hybridkernel/paper/hybridkernel_colm2026.tex": [
            "not a speed result",
            "No throughput, latency, HBM, or GPU speedup claim is made",
            "Kernel logic only; not CUDA or speed evidence",
        ],
        "ssq_lr/paper/ssq_lr_colm2026.tex": [
            "not camera-ready as a method paper until real S1--S3 evidence exists",
            "not model evidence",
            "does not claim weight quantization, activation quantization, KV-cache",
        ],
        "horn/paper/horn_colm2026.tex": [
            "not camera-ready as a method or measurement paper until real H1--H3 evidence exists",
            "flipping its direction label",
            "native systems claim is made yet",
        ],
        "hbsm/paper/hbsm_colm2026.tex": [
            "not camera-ready as a standalone paper until B1--B3",
            "not model evidence",
            "no real forward-sensitivity evidence",
        ],
        "thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex": [
            "diagnostic, not a positive method",
            "no real FP8, CUDA, latency, or throughput result is claimed",
            "failed to reproduce",
        ],
    }
    for relative_path, phrases in required_boundaries.items():
        text = (EXPERIMENTAL / relative_path).read_text()
        for phrase in phrases:
            assert phrase in text, f"{relative_path} is missing boundary phrase: {phrase}"


def test_reviewer_packs_state_camera_readiness_limits_for_active_projects() -> None:
    packs = {
        "hybridkernel/paper/reviewer_pack.md": [
            "not a systems result",
            "does not claim a GPU speedup",
        ],
        "ssq_lr/paper/reviewer_pack.md": [
            "Not camera-ready",
            "needs real S1/S2/S3 tables",
        ],
        "horn/paper/reviewer_pack.md": [
            "Not camera-ready",
            "needs real H1/H2/H3 tables",
        ],
        "hbsm/paper/reviewer_pack.md": [
            "Not camera-ready",
            "needs real B1/B2/B3 evidence",
        ],
        "thoughtflow_fp8/paper/reviewer_pack.md": [
            "ready as a methodology/negative-results workshop diagnostic",
            "no live positive method branch",
        ],
    }
    for relative_path, phrases in packs.items():
        text = (EXPERIMENTAL / relative_path).read_text()
        for phrase in phrases:
            assert phrase in text, f"{relative_path} is missing reviewer limit: {phrase}"
