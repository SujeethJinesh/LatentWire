from __future__ import annotations

import json

from scripts import run_toy_gpa_sparse_dictionary_hub as sweep


def test_toy_gpa_sparse_dictionary_hub_is_deterministic_and_beats_low_shot_fewshot() -> None:
    config = sweep.ToyGPASparseDictionaryHubConfig(
        seed=13,
        dim=18,
        atoms=10,
        classes=5,
        families=5,
        heldout_family=4,
        seen_shots_per_class=20,
        heldout_shots=(1, 2, 4),
        test_examples_per_class=40,
        class_noise=0.18,
        nuisance_rank=4,
        nuisance_strength=0.32,
        source_noise=0.04,
        scale_jitter=0.08,
        bias_scale=0.30,
        style_strength=0.10,
        ridge_lam=1e-2,
        gpa_iters=6,
        dictionary_iters=8,
        topk_atoms=2,
        repair_margin_threshold=0.18,
        repair_margin_gain=0.01,
    )

    payload = sweep.run_experiment(config)
    payload_again = sweep.run_experiment(config)
    assert payload == payload_again
    assert payload["methods"] == list(sweep.METHODS)
    assert list(payload["rows"][0].keys()) == list(sweep.ROW_KEY_ORDER)
    assert len(payload["rows"]) == len(config.heldout_shots) * len(sweep.METHODS)

    for row in payload["rows"]:
        assert row["shot"] in config.heldout_shots
        assert 0.0 <= row["accuracy"] <= 1.0
        assert row["mse"] >= 0.0
        assert -1.0 <= row["accuracy_delta_vs_fewshot"] <= 1.0
        assert row["centroid_cosine"] <= 1.0
        assert row["centroid_cosine"] >= -1.0
        assert row["canonical_gap"] >= 0.0
        if row["method"] in {
            "multiway_gpa_sparse_dictionary",
            "multiway_gpa_sparse_dictionary_repair",
        }:
            assert row["atom_recovery"] is not None
            assert 0.0 <= row["atom_recovery"] <= 1.0
            assert row["dead_atom_rate"] is not None
            assert 0.0 <= row["dead_atom_rate"] <= 1.0
            assert row["codebook_perplexity"] is not None
            assert row["codebook_perplexity"] >= 1.0
        else:
            assert row["atom_recovery"] is None

    lookup = {(row["shot"], row["method"]): row for row in payload["rows"]}
    assert lookup[(1, "multiway_gpa_sparse_dictionary")]["shared_basis"] is True
    assert lookup[(1, "heldout_fewshot_ridge")]["shared_basis"] is False
    assert lookup[(1, "global_seen_ridge")]["accuracy"] < 0.5
    assert lookup[(1, "multiway_gpa_sparse_dictionary")]["mse"] < lookup[(1, "heldout_fewshot_ridge")]["mse"]
    assert lookup[(1, "multiway_gpa_sparse_dictionary")]["mse"] < lookup[(1, "multiway_gpa_canonical")]["mse"]
    assert lookup[(1, "multiway_gpa_sparse_dictionary_repair")]["repair_accept_rate"] >= 0.0


def test_toy_gpa_sparse_dictionary_hub_cli_writes_json_and_markdown(tmp_path) -> None:
    output = tmp_path / "gpa_sparse_dictionary_hub.json"
    markdown = tmp_path / "gpa_sparse_dictionary_hub.md"

    payload = sweep.main(
        [
            "--output",
            str(output),
            "--output-md",
            str(markdown),
            "--seed",
            "13",
            "--dim",
            "18",
            "--atoms",
            "10",
            "--classes",
            "5",
            "--families",
            "5",
            "--heldout-family",
            "4",
            "--seen-shots-per-class",
            "20",
            "--heldout-shots",
            "1",
            "2",
            "4",
            "--test-examples-per-class",
            "24",
            "--class-noise",
            "0.18",
            "--nuisance-rank",
            "4",
            "--nuisance-strength",
            "0.32",
            "--source-noise",
            "0.04",
            "--scale-jitter",
            "0.08",
            "--bias-scale",
            "0.30",
            "--style-strength",
            "0.10",
            "--ridge-lam",
            "0.01",
            "--gpa-iters",
            "6",
            "--dictionary-iters",
            "8",
            "--topk-atoms",
            "2",
            "--repair-margin-threshold",
            "0.18",
            "--repair-margin-gain",
            "0.01",
        ]
    )

    loaded = json.loads(output.read_text())
    assert loaded == payload
    text = markdown.read_text()
    assert "# Toy GPA Sparse Dictionary Hub Sweep" in text
    assert "multiway_gpa_sparse_dictionary" in text
    assert "https://arxiv.org/abs/2502.03714" in text
