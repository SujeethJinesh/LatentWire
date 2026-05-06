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
        assert (root / "paper/reviewer_pack.md").exists()


def test_active_project_reviewer_packs_cover_colm_axes() -> None:
    expected = {
        "ssq_lr": ["S1--S3", "prefill_end", "at least 12 prompts"],
        "horn": ["H1--H3", "both boundary directions", "matched flipped controls"],
        "hbsm": ["B1--B3", "perturbation-off", "existing sensitivity tools"],
    }
    for project, project_phrases in expected.items():
        pack = (EXPERIMENTAL / project / "paper/reviewer_pack.md").read_text()
        for axis in ["Benchmarks", "Ablations", "Correctness", "Reproducibility", "Novelty", "Camera-readiness"]:
            assert f"| {axis} |" in pack
        for phrase in project_phrases:
            assert phrase in pack


def test_active_project_paper_shells_state_real_packet_blockers() -> None:
    required = {
        "ssq_lr/paper/ssq_lr_colm2026.tex": [
            "not camera-ready as a method paper until real S1--S3 evidence exists",
            "\\texttt{prefill\\_end}",
            "at least 12 prompt IDs",
        ],
        "horn/paper/horn_colm2026.tex": [
            "not camera-ready as a method or measurement paper until real H1--H3 evidence exists",
            "cover both \\texttt{attention->ssm} and",
            "flipping its direction",
        ],
        "hbsm/paper/hbsm_colm2026.tex": [
            "not camera-ready as a standalone paper until B1--B3",
            "perturbation-off row whose drift is",
            "both true and false boundary flags",
        ],
        "hybridkernel/paper/hybridkernel_colm2026.tex": [
            "no\\_boundary\\_signal\\_kill",
            "missing or external trace references are rejected",
        ],
    }
    for relative_path, phrases in required.items():
        text = (EXPERIMENTAL / relative_path).read_text()
        for phrase in phrases:
            assert phrase in text, f"{relative_path} missing {phrase}"


def test_killed_markers_are_one_page_decision_records() -> None:
    for readme_path in EXPERIMENTAL.glob("KILLED_*/README.md"):
        text = readme_path.read_text()
        for phrase in ["What Was Tried", "Why It Died", "Salvage Value"]:
            assert phrase in text, f"{readme_path} missing {phrase}"
