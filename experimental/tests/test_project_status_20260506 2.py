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


def test_hybrid_quant_later_gates_are_not_claimed_as_current_evidence() -> None:
    expected = {
        "ssq_lr": ["S2/S3 now have follow-up contract checks", "not current evidence"],
        "horn": ["H2/H3 now have follow-up contract checks", "not current evidence"],
        "hbsm": ["B2/B3 now have follow-up contract checks", "not current evidence"],
    }

    for project, phrases in expected.items():
        readme = (EXPERIMENTAL / project / "README.md").read_text()
        pack = (EXPERIMENTAL / project / "paper/reviewer_pack.md").read_text()
        paper = next((EXPERIMENTAL / project / "paper").glob("*.tex")).read_text()
        for phrase in phrases:
            assert phrase in readme or phrase in pack or phrase in paper


def test_active_project_paper_shells_state_real_packet_blockers() -> None:
    required = {
        "ssq_lr/paper/ssq_lr_colm2026.tex": [
            "not camera-ready as a method paper until non-resource-limited, passing S1--S3 evidence exists",
            "\\texttt{prefill\\_end}",
            "at least 12 prompt IDs",
        ],
        "horn/paper/horn_colm2026.tex": [
            "not camera-ready as a method or measurement paper until real H1--H3 evidence exists",
            "cover both \\texttt{attention->ssm} and",
            "flipping its direction",
        ],
        "hbsm/paper/hbsm_colm2026.tex": [
            "current HBSM hypothesis is demoted",
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


def test_native_gpu_handoff_has_strict_active_project_commands() -> None:
    handoff = (EXPERIMENTAL / "native_gpu_handoff_20260506.md").read_text()

    for project in ["HybridKernel", "SSQ-LR", "HORN", "HBSM", "ThoughtFlow-FP8"]:
        assert project in handoff
    assert "--require-full-matrix" in handoff
    assert "--require-token-counts" in handoff
    for project in ["ssq_lr", "horn", "hbsm"]:
        assert f"--project {project}" in handoff
    assert "TO_FILL_BEFORE_CAPTURE" in handoff
    assert "Only after a real HBSM B1 packet establishes sensitivity heterogeneity" in handoff


def test_hybridkernel_gpu_runbook_pins_post_promotion_quality_smoke() -> None:
    runbook = (EXPERIMENTAL / "hybridkernel/phase2/nvidia_vllm_profiler_runbook.md").read_text()
    readme = (EXPERIMENTAL / "hybridkernel/README.md").read_text()

    for text in [runbook, readme]:
        assert "hybrid_reasoning_smoke_12_20260506.jsonl" in text
        assert "zero normalized" in text
        assert "exact-answer" in text
        assert "regressions" in text
        assert "mean output-length drift within" in text
        assert "10%" in text


def test_hybridkernel_paper_tracks_strict_gpu_packet_provenance() -> None:
    paper = (EXPERIMENTAL / "hybridkernel/paper/hybridkernel_colm2026.tex").read_text()
    pack = (EXPERIMENTAL / "hybridkernel/paper/reviewer_pack.md").read_text()

    for text in [paper, pack]:
        assert "environment.json" in text
        assert "snapshot manifest" in text.lower() or "snapshot manifests" in text.lower()
        assert "control_window_ids" in text or "control\\_window\\_ids" in text


def test_killed_markers_are_one_page_decision_records() -> None:
    for readme_path in EXPERIMENTAL.glob("KILLED_*/README.md"):
        text = readme_path.read_text()
        for phrase in ["What Was Tried", "Why It Died", "Salvage Value"]:
            assert phrase in text, f"{readme_path} missing {phrase}"
