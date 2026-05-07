from pathlib import Path


PAPER = Path(__file__).resolve().parents[2] / "paper/thoughtflow_fp8_colm2026.tex"


def test_paper_uses_preregistered_paired_interval_gate_language() -> None:
    paper = PAPER.read_text(encoding="utf-8")

    assert "Promotion requires the preregistered mean margin and paired CI high below zero." in paper
    assert "Mean margin plus paired CI high must be below zero." not in paper


def test_paper_marks_saved_distilgpt2_surfaces_as_falsification_fixtures() -> None:
    paper = PAPER.read_text(encoding="utf-8")

    assert "\\texttt{distilgpt2} saved continuations" in paper
    assert "not reasoning-model benchmarks" in paper
    assert "not faithful external implementations" in paper


def test_paper_has_stable_owned_repro_command_and_local_triton_note() -> None:
    paper = PAPER.read_text(encoding="utf-8")

    assert "TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1" in paper
    assert "./venv_arm64/bin/python -m pytest" in paper
    assert "experimental/thoughtflow_fp8/phase2/tests" in paper
    assert "experimental/thoughtflow_fp8/phase4/tests" in paper
    assert "experimental/thoughtflow_fp8/phase4/triton_interpreter_note_20260506.md" in paper
    assert "experimental/triton_cpu_source_install_20260506.md" not in paper
