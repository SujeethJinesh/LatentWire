from pathlib import Path

def test_plan_contains_sections():
    plan_text = Path("PLAN.md").read_text()
    assert "LatentWire Phase-2 Accuracy Plan" in plan_text
    assert "Milestone" in plan_text or "Item" in plan_text
