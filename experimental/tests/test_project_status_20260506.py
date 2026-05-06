from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL = ROOT / "experimental"


def test_project_status_tracks_active_and_killed_branches() -> None:
    status = (EXPERIMENTAL / "project_status_20260506.md").read_text()

    for project in ["HybridKernel", "SSQ-LR", "HORN", "HBSM", "ThoughtFlow-FP8"]:
        assert project in status
    assert "SinkKV" not in status

    for marker in [
        "KILLED_sinkaware_static_prior",
        "KILLED_sinkaware_systems_framing",
        "KILLED_thoughtflow_fp8_positive_method",
        "KILLED_anchorspec",
        "KILLED_phasequant",
        "KILLED_moe_phase_routing",
    ]:
        assert marker in status
        assert (EXPERIMENTAL / marker / "README.md").exists()


def test_new_project_scaffolds_have_preregistrations_and_colm_shells() -> None:
    expected = {
        "ssq_lr": "phase2/preregister_ssq_lr_20260506.md",
        "horn": "phase2/preregister_horn_20260506.md",
        "hbsm": "phase2/preregister_hbsm_20260506.md",
    }

    for project, prereg in expected.items():
        root = EXPERIMENTAL / project
        assert (root / "README.md").exists()
        assert (root / "progress.md").exists()
        assert (root / prereg).exists()
        assert (root / "phase1/competitor_matrix.md").exists()
        assert next((root / "paper").glob("*.tex")).exists()
        assert next((root / "paper").glob("*.pdf")).exists()


def test_killed_markers_are_one_page_decision_records() -> None:
    for readme_path in EXPERIMENTAL.glob("KILLED_*/README.md"):
        text = readme_path.read_text()
        for phrase in ["What Was Tried", "Why It Died", "Salvage Value"]:
            assert phrase in text, f"{readme_path} missing {phrase}"
