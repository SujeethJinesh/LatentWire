from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL = ROOT / "experimental"


def test_top_level_readiness_index_records_current_stop_conditions() -> None:
    readme = (EXPERIMENTAL / "README.md").read_text()
    assert "Current Mac-Complete Status" in readme
    assert "Weakly alive as a native-profiler handoff only" in readme
    assert "Alive but bounded as an approximate rank-2 sink-logit branch" in readme
    assert "Stopped as a positive method" in readme
    assert "not completed GPU systems papers" in readme


def test_mac_complete_audit_links_all_project_packets() -> None:
    audit = (EXPERIMENTAL / "mac_complete_readiness_20260506.md").read_text()
    for relative_path in [
        "experimental/hybridkernel/paper/reviewer_pack.md",
        "experimental/sinkaware/paper/reviewer_pack.md",
        "experimental/thoughtflow_fp8/paper/reviewer_pack.md",
        "experimental/native_gpu_handoff_20260506.md",
        "experimental/hybridkernel/paper/hybridkernel_colm2026.pdf",
        "experimental/sinkaware/paper/sinkaware_colm2026.pdf",
        "experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf",
    ]:
        assert relative_path in audit
        assert (ROOT / relative_path).exists()


def test_native_handoff_map_has_project_gates_and_no_thoughtflow_gpu_work() -> None:
    handoff = (EXPERIMENTAL / "native_gpu_handoff_20260506.md").read_text()
    for phrase in [
        "SinkAware rank-2 native timing",
        "HybridKernel profiler packet",
        "ThoughtFlow-FP8",
        "experimental/sinkaware/phase2/check_native_gpu_packet.py",
            "no GPU work for the current branch set",
        "at least a 3% native speed or memory-traffic improvement",
        "recoverable-gain upper bound clears",
    ]:
        assert phrase in handoff


def test_colm_style_pdfs_are_present_and_bounded() -> None:
    expected_pages = {
        "hybridkernel/paper/hybridkernel_colm2026.pdf": 3,
        "sinkaware/paper/sinkaware_colm2026.pdf": 5,
        "thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf": 5,
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
        "sinkaware/paper/sinkaware_colm2026.tex": [
            "not cross-model predictor transfer, benchmark success, or a GPU speed result",
            "no benchmark accuracy or GPU speed is measured",
            "not evidence of GPU performance",
            "passes \\texttt{check\\_native\\_gpu\\_packet.py}",
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


def test_reviewer_packs_state_camera_readiness_limits() -> None:
    packs = {
        "hybridkernel/paper/reviewer_pack.md": [
            "not a systems result",
            "does not claim a GPU speedup",
        ],
        "sinkaware/paper/reviewer_pack.md": [
            "alive but bounded",
            "No GPU latency or memory claim exists yet",
        ],
        "thoughtflow_fp8/paper/reviewer_pack.md": [
            "ready only as a negative/mixed workshop diagnostic",
            "no live positive method branch",
        ],
    }
    for relative_path, phrases in packs.items():
        text = (EXPERIMENTAL / relative_path).read_text()
        for phrase in phrases:
            assert phrase in text, f"{relative_path} is missing reviewer limit: {phrase}"
