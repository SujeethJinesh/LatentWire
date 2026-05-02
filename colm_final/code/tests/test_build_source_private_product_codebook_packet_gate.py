from __future__ import annotations

from scripts.build_source_private_product_codebook_packet_gate import build_product_codebook_gate


def test_product_codebook_packet_gate_writes_aggregate(tmp_path) -> None:
    payload = build_product_codebook_gate(
        output_dir=tmp_path / "product_codebook",
        remap_seeds=[101],
        budgets=[2, 4],
        train_examples=64,
        eval_examples=32,
        feature_dim=128,
        train_seed=5,
        eval_seed=6,
    )

    assert payload["gate"] == "source_private_product_codebook_packet_gate"
    assert payload["headline"]["rows"] == 2
    assert "product_codebook_accuracy" in payload["rows"][0]
    assert "product_codebook_minus_best_control" in payload["rows"][0]
    assert "p50_decode_latency_ms" in payload["rows"][0]
    assert (tmp_path / "product_codebook" / "product_codebook_packet_gate.md").exists()
    assert (tmp_path / "product_codebook" / "remap_101" / "summary.json").exists()
