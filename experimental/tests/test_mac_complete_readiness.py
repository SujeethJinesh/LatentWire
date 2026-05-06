from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTAL = ROOT / "experimental"


def test_top_level_readiness_index_records_current_stop_conditions() -> None:
    readme = (EXPERIMENTAL / "README.md").read_text()
    assert "Experimental Project Control Plane" in readme
    assert "HybridKernel" in readme
    assert "SSQ-LR" in readme
    assert "HORN" in readme
    assert "HBSM" in readme
    assert "ThoughtFlow-FP8" in readme
    assert "These utilities support Mac-local hypothesis gates" in readme


def test_mac_complete_audit_links_all_project_packets() -> None:
    audit = (EXPERIMENTAL / "mac_complete_readiness_20260506.md").read_text()
    for relative_path in [
        "experimental/hybridkernel/paper/reviewer_pack.md",
        "experimental/sinkaware/paper/reviewer_pack.md",
        "experimental/thoughtflow_fp8/paper/reviewer_pack.md",
        "experimental/sinkaware/phase0/setup_complete.md",
        "experimental/native_gpu_handoff_20260506.md",
        "experimental/hybridkernel/paper/hybridkernel_colm2026.pdf",
        "experimental/sinkaware/paper/sinkaware_colm2026.pdf",
        "experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf",
    ]:
        assert relative_path in audit
        assert (ROOT / relative_path).exists()


def test_final_saturation_review_records_all_stop_decisions() -> None:
    audit = (EXPERIMENTAL / "mac_complete_readiness_20260506.md").read_text()
    for phrase in [
        "All three returned `STOP`",
        "| HybridKernel | `STOP` |",
        "| SinkAware | `STOP` after setup closure |",
        "| ThoughtFlow-FP8 | `STOP` |",
        "no SSH, and no NVIDIA GPU",
    ]:
        assert phrase in audit


def test_sinkaware_setup_complete_records_current_venv_requirements() -> None:
    setup = (EXPERIMENTAL / "sinkaware/phase0/setup_complete.md").read_text()
    for phrase in [
        "repo-root `./venv_arm64`",
        "No broken requirements found",
        "IPython",
        "einops",
        "tiktoken",
        "triton==3.7.0+git270e696d",
        "native NVIDIA packet",
    ]:
        assert phrase in setup


def test_sinkaware_stop_gate_is_machine_guarded() -> None:
    progress = (EXPERIMENTAL / "sinkaware/progress.md").read_text()
    audit = (EXPERIMENTAL / "mac_complete_readiness_20260506.md").read_text()
    runbook = (EXPERIMENTAL / "sinkaware/phase2/gpu_gate_runbook.md").read_text()

    for phrase in [
        "NO ADDITIONAL MAC-SIDE SINKAWARE EXPERIMENT REMAINS",
        "blocked on native NVIDIA quality/timing/memory evidence only",
    ]:
        assert phrase in progress

    for phrase in [
        "STOP` after setup closure",
        "phase0/setup_complete.md",
    ]:
        assert phrase in audit

    for phrase in [
        "top-1 disagreement aggregated over all measured rows",
        "model/shape subgroup values",
        "check_native_gpu_packet.py",
    ]:
        assert phrase in runbook


def test_thoughtflow_reopen_gate_is_machine_guarded() -> None:
    audit = (EXPERIMENTAL / "mac_complete_readiness_20260506.md").read_text()
    manifest = (EXPERIMENTAL / "thoughtflow_fp8/phase2/current_decision_manifest_20260506.md").read_text()
    reviewer_pack = (EXPERIMENTAL / "thoughtflow_fp8/paper/reviewer_pack.md").read_text()

    for phrase in [
        "fresh/larger frozen sparse-cache surface",
        "matched-budget quality wins",
        "same/cross-family separation",
        "paired uncertainty",
        "oracle/headroom diagnostics",
    ]:
        assert phrase in audit

    for phrase in [
        "STOP / diagnostic only",
        "one-shot evaluation only on a fresh/larger frozen sparse-cache surface",
    ]:
        assert phrase in manifest

    for phrase in [
        "no live positive method branch",
        "new pre-registered utility signal",
    ]:
        assert phrase in reviewer_pack


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
        "thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf": 6,
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


def test_hybridkernel_stop_gate_is_machine_guarded() -> None:
    paper = (EXPERIMENTAL / "hybridkernel/paper/hybridkernel_colm2026.tex").read_text()
    readme = (EXPERIMENTAL / "hybridkernel/README.md").read_text()
    progress = (EXPERIMENTAL / "hybridkernel/progress.md").read_text()
    runbook = (EXPERIMENTAL / "hybridkernel/phase2/nvidia_vllm_profiler_runbook.md").read_text()

    for phrase in [
        "No native NVIDIA/vLLM profile has been run",
        "owned Mac suite passes with one opt-in CPU-backend test skipped by default",
    ]:
        assert phrase in paper
    assert "colm_workshop_scaffold.md" not in paper

    for phrase in [
        "Local Mac work is saturated",
        "./venv_arm64",
        "repo-local `triton-cpu` source build",
        "add more local kernels, scaffolds, or paper claims",
    ]:
        assert phrase in readme

    for phrase in [
            "NO ADDITIONAL MAC KERNEL OR BENCHMARK WORK REMAINS",
            "native server-side Nsight evidence or kill/shelve",
        "check_profiler_run_artifacts.py",
        "analyze_profiler_metrics.py",
        "KILLED_mac_only_kernel_iteration",
    ]:
        assert phrase in progress

    for phrase in [
        "server-side Nsight Systems and Nsight Compute",
        "dry_run: false",
        "check_profiler_run_artifacts.py",
    ]:
        assert phrase in runbook


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
            "ready as a methodology/negative-results workshop diagnostic",
            "no live positive method branch",
        ],
    }
    for relative_path, phrases in packs.items():
        text = (EXPERIMENTAL / relative_path).read_text()
        for phrase in phrases:
            assert phrase in text, f"{relative_path} is missing reviewer limit: {phrase}"
