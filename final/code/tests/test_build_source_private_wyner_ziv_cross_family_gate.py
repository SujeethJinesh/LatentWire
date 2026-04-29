from __future__ import annotations

from scripts.build_source_private_wyner_ziv_cross_family_gate import build_cross_family_gate


def test_wyner_ziv_cross_family_gate_writes_directions(tmp_path) -> None:
    payload = build_cross_family_gate(
        output_dir=tmp_path / "cross",
        budgets=[2],
        train_examples=32,
        eval_examples=16,
        feature_dim=64,
        seed=5,
    )

    assert payload["gate"] == "source_private_wyner_ziv_cross_family_gate"
    assert {row["direction"] for row in payload["rows"]} == {"core_to_holdout", "holdout_to_core"}
    assert len(payload["rows"]) == 2
    assert (tmp_path / "cross" / "wyner_ziv_cross_family_gate.md").exists()
