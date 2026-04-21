from __future__ import annotations

import json

from scripts import run_toy_tokenizer_bridge as bridge


def test_toy_tokenizer_bridge_rows_are_deterministic_and_sane() -> None:
    config = bridge.ToyTokenizerBridgeConfig(
        seed=7,
        examples=48,
        min_terms=3,
        max_terms=5,
        byte_noise_rate=1.0,
        span_noise_rate=1.0,
    )
    source, target = bridge._build_tokenizers()

    rows = bridge._evaluate_methods(config, source=source, target=target)
    rows_again = bridge._evaluate_methods(config, source=source, target=target)

    assert rows == rows_again
    assert [row["method"] for row in rows] == [
        "token_id",
        "vocab_remap",
        "byte_span_canonical",
        "byte_span_noisy_bytes",
        "byte_span_noisy_spans",
    ]

    lookup = {row["method"]: row for row in rows}
    canonical = lookup["byte_span_canonical"]
    noisy_bytes = lookup["byte_span_noisy_bytes"]
    noisy_spans = lookup["byte_span_noisy_spans"]
    token_id = lookup["token_id"]
    vocab_remap = lookup["vocab_remap"]

    assert canonical["exact_reconstruction"] == 1.0
    assert canonical["digit_accuracy"] == 1.0
    assert canonical["operator_accuracy"] == 1.0
    assert noisy_bytes["exact_reconstruction"] < 1.0
    assert noisy_spans["exact_reconstruction"] < 1.0
    assert 0.0 <= token_id["exact_reconstruction"] <= 1.0
    assert 0.0 <= vocab_remap["exact_reconstruction"] <= 1.0
    assert vocab_remap["exact_reconstruction"] >= token_id["exact_reconstruction"]
    for row in rows:
        assert 0.0 <= row["digit_accuracy"] <= 1.0
        assert 0.0 <= row["operator_accuracy"] <= 1.0
        assert row["fragmentation_rate"] > 0.0
        assert row["bytes_per_example"] > 0.0


def test_toy_tokenizer_bridge_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "toy_tokenizer.json"
    markdown = tmp_path / "toy_tokenizer.md"

    payload = bridge.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "11",
            "--examples",
            "24",
            "--min-terms",
            "3",
            "--max-terms",
            "4",
            "--byte-noise-rate",
            "1.0",
            "--span-noise-rate",
            "1.0",
        ]
    )

    on_disk = json.loads(output.read_text())
    assert on_disk == payload
    assert on_disk["config"]["seed"] == 11
    assert on_disk["config"]["examples"] == 24
    assert len(on_disk["rows"]) == 5
    assert on_disk["rows"][2]["method"] == "byte_span_canonical"

    md = markdown.read_text()
    assert "# Toy Tokenizer Bridge" in md
    assert "| Method | Exact recon | Digit acc | Operator acc | Fragmentation | Bytes/example |" in md
