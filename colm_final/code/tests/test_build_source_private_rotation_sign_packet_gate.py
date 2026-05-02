from __future__ import annotations

from scripts.build_source_private_rotation_sign_packet_gate import build_rotation_sign_gate


def test_rotation_sign_packet_gate_writes_aggregate(tmp_path) -> None:
    payload = build_rotation_sign_gate(
        output_dir=tmp_path / "rotation_sign",
        remap_seeds=[101],
        budgets=[2, 4],
        train_examples=64,
        eval_examples=32,
        feature_dim=128,
        train_seed=5,
        eval_seed=6,
    )

    assert payload["gate"] == "source_private_rotation_sign_packet_gate"
    assert payload["headline"]["rows"] == 2
    assert "rotation_sign_accuracy" in payload["rows"][0]
    assert "rotation_sign_minus_best_control" in payload["rows"][0]
    assert "p50_decode_latency_ms" in payload["rows"][0]
    assert (tmp_path / "rotation_sign" / "rotation_sign_packet_gate.md").exists()
    assert (tmp_path / "rotation_sign" / "remap_101" / "summary.json").exists()
