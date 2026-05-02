from __future__ import annotations

from scripts.build_anchor_relative_sparse_packet_gate import build_anchor_relative_sparse_gate


def test_anchor_relative_sparse_gate_writes_bidirectional_rows(tmp_path) -> None:
    payload = build_anchor_relative_sparse_gate(
        output_dir=tmp_path / "anchor_sparse",
        budgets=[2],
        train_examples=32,
        eval_examples=16,
        feature_dim=64,
        seed=5,
        candidate_view="semantic",
        ridge=1e-2,
    )

    assert payload["gate"] == "anchor_relative_sparse_packet_gate"
    assert {row["direction"] for row in payload["rows"]} == {"core_to_holdout", "holdout_to_core"}
    assert len(payload["rows"]) == 2
    assert payload["headline"]["budgets"] == [2]
    assert (tmp_path / "anchor_sparse" / "anchor_relative_sparse_packet_gate.md").exists()
    assert (tmp_path / "anchor_sparse" / "core_to_holdout" / "predictions_budget2.jsonl").exists()
