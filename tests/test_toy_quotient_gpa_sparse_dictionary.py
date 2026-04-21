from __future__ import annotations

import json

from scripts import run_toy_quotient_gpa_sparse_dictionary as sweep


def test_toy_quotient_gpa_sparse_dictionary_is_deterministic_and_has_expected_boundaries() -> None:
    config = sweep.ToyQuotientGPASparseDictionaryConfig(
        seed=7,
        heads=4,
        head_dim=6,
        atoms=10,
        classes=5,
        families=5,
        heldout_family=4,
        anchor_family=0,
        seen_shots_per_class=20,
        heldout_shots=(1, 2, 4),
        test_examples_per_class=40,
        class_noise=0.16,
        source_noise=0.03,
        nuisance_rank=3,
        nuisance_strength=0.22,
        head_scale_jitter=0.18,
        head_bias_scale=0.18,
        ridge_lam=1e-2,
        gpa_iters=8,
        dictionary_iters=10,
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
        assert -1.0 <= row["centroid_cosine"] <= 1.0
        assert row["gauge_residual"] >= 0.0
        if row["shared_basis"]:
            assert row["head_match_accuracy"] is not None
            assert 0.0 <= row["head_match_accuracy"] <= 1.0
            assert row["canonical_gap"] is not None
            assert row["canonical_gap"] >= 0.0
        if "sparse_dictionary" in row["method"]:
            assert row["atom_recovery"] is not None
            assert 0.0 <= row["atom_recovery"] <= 1.0
            assert row["dead_atom_rate"] is not None
            assert 0.0 <= row["dead_atom_rate"] <= 1.0
            assert row["codebook_perplexity"] is not None
            assert row["codebook_perplexity"] >= 1.0

    lookup = {(row["shot"], row["method"]): row for row in payload["rows"]}

    assert lookup[(1, "quotient_match_after_fix")]["head_match_accuracy"] == 1.0
    assert lookup[(1, "quotient_gpa_canonical")]["head_match_accuracy"] == 1.0
    assert lookup[(1, "quotient_gpa_sparse_dictionary")]["head_match_accuracy"] == 1.0

    assert lookup[(1, "quotient_gpa_sparse_dictionary")]["mse"] <= lookup[(1, "quotient_match_after_fix")]["mse"]
    assert lookup[(1, "quotient_gpa_sparse_dictionary")]["mse"] <= lookup[(1, "quotient_gpa_canonical")]["mse"]
    assert lookup[(1, "quotient_gpa_sparse_dictionary")]["mse"] <= lookup[(1, "heldout_fewshot_ridge")]["mse"]

    assert lookup[(2, "quotient_gpa_sparse_dictionary")]["mse"] <= lookup[(2, "heldout_fewshot_ridge")]["mse"]
    assert lookup[(4, "heldout_fewshot_ridge")]["mse"] <= lookup[(4, "quotient_gpa_sparse_dictionary")]["mse"]

    assert lookup[(1, "quotient_gpa_sparse_dictionary_repair")]["repair_accept_rate"] == 0.0
    assert lookup[(1, "quotient_gpa_sparse_dictionary_repair")]["repair_help_rate"] == 0.0
    assert lookup[(1, "quotient_gpa_sparse_dictionary_repair")]["repair_harm_rate"] == 0.0


def test_toy_quotient_gpa_sparse_dictionary_cli(tmp_path) -> None:
    output_path = tmp_path / "quotient_gpa_sparse_dictionary.json"
    markdown_path = tmp_path / "quotient_gpa_sparse_dictionary.md"

    payload = sweep.main(
        [
            "--output",
            str(output_path),
            "--output-md",
            str(markdown_path),
            "--seed",
            "7",
            "--heads",
            "4",
            "--head-dim",
            "6",
            "--atoms",
            "10",
            "--classes",
            "5",
            "--families",
            "5",
            "--heldout-family",
            "4",
            "--anchor-family",
            "0",
            "--seen-shots-per-class",
            "20",
            "--heldout-shots",
            "1",
            "2",
            "4",
            "--test-examples-per-class",
            "40",
            "--class-noise",
            "0.16",
            "--source-noise",
            "0.03",
            "--nuisance-rank",
            "3",
            "--nuisance-strength",
            "0.22",
            "--head-scale-jitter",
            "0.18",
            "--head-bias-scale",
            "0.18",
            "--ridge-lam",
            "0.01",
            "--gpa-iters",
            "8",
            "--dictionary-iters",
            "10",
            "--topk-atoms",
            "2",
            "--repair-margin-threshold",
            "0.18",
            "--repair-margin-gain",
            "0.01",
        ]
    )

    saved = json.loads(output_path.read_text())
    assert saved == payload
    markdown = markdown_path.read_text()
    assert "Toy Quotient + GPA Sparse Dictionary Sweep" in markdown
    assert "quotient_gpa_sparse_dictionary" in markdown
    assert "Multi-Way Representation Alignment" in markdown
