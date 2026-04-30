from __future__ import annotations

from scripts.build_source_private_product_codebook_decode_frontier import build_decode_frontier


def test_product_codebook_decode_frontier_matches_canonical_decoder(tmp_path) -> None:
    payload = build_decode_frontier(
        output_dir=tmp_path / "decode_frontier",
        product_gate_json=None,
        remap_seeds=[101],
        budgets=[2],
        train_examples=64,
        eval_examples=32,
        feature_dim=128,
        train_seed=5,
        eval_seed=6,
        candidate_view="slot",
        timing_repeats=1,
        batch_repeats=2,
    )

    assert payload["gate"] == "source_private_product_codebook_decode_frontier"
    assert payload["headline"]["rows"] == 1
    assert payload["rows"][0]["prediction_mismatch_count"] == 0
    assert payload["rows"][0]["cached_receiver_p50_ms"] >= 0.0
    assert "cached_speedup_vs_cold_receiver" in payload["rows"][0]
    assert (tmp_path / "decode_frontier" / "product_codebook_decode_frontier.json").exists()
    assert (tmp_path / "decode_frontier" / "manifest.json").exists()
