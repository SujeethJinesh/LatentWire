from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_consumed_preregistrations_are_closed() -> None:
    for relative in [
        "ssq_lr/phase2/preregister_ssq_lr_20260506.md",
        "horn/phase2/preregister_horn_20260506.md",
        "hbsm/phase2/preregister_hbsm_20260506.md",
    ]:
        text = _read(relative)
        assert "Closure: this preregistration was consumed" in text
        assert "any reopening requires a new" in text
        assert "preregistration before collecting rows" in text


def test_horn_h2_resource_limited_decision_is_not_promotable() -> None:
    required = "RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION"
    for relative in [
        "horn/README.md",
        "horn/progress.md",
        "horn/paper/reviewer_pack.md",
        "horn/paper/horn_colm2026.tex",
    ]:
        text = _read(relative)
        normalized = text.replace("\\_", "_")
        assert required in normalized
        assert "raw gate status" in text or "raw `gate_status`" in text


def test_ssq_lr_progress_has_no_post_stop_live_wording() -> None:
    text = _read("ssq_lr/progress.md")
    for banned in [
        "current positive Mac signal",
        "S2 WAS REVIVED",
        "SSQ-LR IS ALIVE",
        "Selected/live recipe",
    ]:
        assert banned not in text
