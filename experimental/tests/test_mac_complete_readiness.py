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
        "experimental/hybridkernel/paper/hybridkernel_colm2026.pdf",
        "experimental/sinkaware/paper/sinkaware_colm2026.pdf",
        "experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf",
    ]:
        assert relative_path in audit
        assert (ROOT / relative_path).exists()


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
