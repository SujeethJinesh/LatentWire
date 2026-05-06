from pathlib import Path


PAPER = Path(__file__).resolve().parents[2] / "paper/thoughtflow_fp8_colm2026.tex"


def test_paper_uses_preregistered_paired_interval_gate_language() -> None:
    paper = PAPER.read_text(encoding="utf-8")

    assert "Promotion requires the preregistered mean margin and paired CI high below zero." in paper
    assert "Mean margin plus paired CI high must be below zero." not in paper
