from __future__ import annotations

from scripts.build_source_private_wyner_ziv_packet_gate import build_wyner_ziv_gate


def test_wyner_ziv_packet_gate_writes_aggregate(tmp_path) -> None:
    payload = build_wyner_ziv_gate(
        output_dir=tmp_path / "wz",
        remap_seeds=[101],
        budgets=[2, 4],
        train_examples=64,
        eval_examples=32,
        feature_dim=128,
        train_seed=5,
        eval_seed=6,
    )

    assert payload["gate"] == "source_private_wyner_ziv_packet_gate"
    assert payload["headline"]["rows"] == 2
    assert payload["rows"][0]["query_aware_oracle_bytes"] == 14.0
    assert "scalar_wyner_ziv_accuracy" in payload["rows"][0]
    assert (tmp_path / "wz" / "wyner_ziv_packet_gate.md").exists()
    assert (tmp_path / "wz" / "remap_101" / "summary.json").exists()
