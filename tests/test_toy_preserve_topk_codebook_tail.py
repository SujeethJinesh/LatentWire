from __future__ import annotations

import json

from scripts import run_toy_preserve_topk_codebook_tail as codec


def test_preserve_topk_codebook_tail_returns_deterministic_rows() -> None:
    config = codec.ToyPreserveTopKCodebookTailConfig(
        seed=7,
        calibration_samples=64,
        test_samples=72,
        atoms=16,
        signal_atoms=4,
        outlier_atoms=2,
        preserved_atoms=4,
        codebook_size=6,
        codebook_iters=6,
    )

    payload = codec.run_experiment(config)
    rows = {row["method"]: row for row in payload["rows"]}

    assert set(rows) == {
        "uniform_low_bit",
        "preserve_topk_uniform_tail",
        "preserve_topk_codebook_tail",
        "preserve_topk_codebook_tail_residual_fix",
    }
    assert rows["preserve_topk_uniform_tail"]["mse"] <= rows["uniform_low_bit"]["mse"]
    assert rows["preserve_topk_codebook_tail_residual_fix"]["mse"] <= rows["preserve_topk_codebook_tail"]["mse"]
    assert rows["preserve_topk_codebook_tail"]["codebook_perplexity"] is not None

    payload_again = codec.run_experiment(config)
    assert payload_again == payload


def test_preserve_topk_codebook_tail_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_codec.json"
    markdown = tmp_path / "toy_codec.md"

    codec.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "5",
            "--calibration-samples",
            "32",
            "--test-samples",
            "40",
            "--atoms",
            "16",
            "--signal-atoms",
            "4",
            "--outlier-atoms",
            "2",
            "--preserved-atoms",
            "4",
            "--codebook-size",
            "6",
            "--codebook-iters",
            "4",
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["config"]["atoms"] == 16
    assert len(payload["rows"]) == 4
    assert "Toy Preserve-TopK Codebook Tail" in markdown.read_text()
