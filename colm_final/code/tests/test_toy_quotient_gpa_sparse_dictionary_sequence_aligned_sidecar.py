from __future__ import annotations

import json

from scripts import run_toy_quotient_gpa_sparse_dictionary_sequence_aligned_sidecar as sweep


def test_sequence_aligned_sidecar_is_deterministic_and_improves_over_plain_sidecar() -> None:
    config = sweep.ToyQuotientGPASparseDictionarySequenceAlignedSidecarConfig(
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
        remap_capacity=10,
        interface_strength=2.5,
        remap_recovery=0.7,
        sidecar_hash_dim=16,
        sidecar_ngram_max=2,
        sidecar_scale=0.35,
        alignment_hash_dim=16,
        alignment_scale=0.30,
        alignment_profile_scale=0.20,
        alignment_token_scale=0.75,
    )

    payload = sweep.run_experiment(config)
    payload_again = sweep.run_experiment(config)
    assert payload == payload_again
    assert payload["methods"] == list(sweep.METHODS)
    assert list(payload["rows"][0].keys()) == list(sweep.ROW_KEY_ORDER)
    assert len(payload["rows"]) == len(config.heldout_shots) * len(sweep.METHODS)

    lookup = {(row["shot"], row["method"]): row for row in payload["rows"]}
    for shot in config.heldout_shots:
        assert lookup[(shot, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id")]["shared_basis"] is True
        assert lookup[(shot, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap")]["shared_basis"] is True
        assert lookup[(shot, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id")]["head_match_accuracy"] == 1.0
        assert lookup[(shot, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap")]["head_match_accuracy"] == 1.0
        assert lookup[(shot, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id")]["mean_alignment_sidecar_norm"] is not None
        assert lookup[(shot, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap")]["mean_alignment_sidecar_norm"] is not None

    assert lookup[(1, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap")]["mse"] < lookup[(1, "quotient_gpa_sparse_dictionary_byte_sidecar_remap")]["mse"]
    assert lookup[(2, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap")]["mse"] < lookup[(2, "quotient_gpa_sparse_dictionary_byte_sidecar_remap")]["mse"]
    assert lookup[(1, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id")]["mse"] < lookup[(1, "quotient_gpa_sparse_dictionary_byte_sidecar_token_id")]["mse"]
    assert lookup[(2, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_token_id")]["mse"] < lookup[(2, "quotient_gpa_sparse_dictionary_byte_sidecar_token_id")]["mse"]
    assert lookup[(1, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap")]["mse"] < lookup[(1, "heldout_fewshot_ridge_token_id")]["mse"]
    assert lookup[(2, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap")]["mse"] < lookup[(2, "heldout_fewshot_ridge_token_id")]["mse"]
    assert lookup[(4, "heldout_fewshot_ridge_byte_span_remap")]["mse"] < lookup[(4, "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap")]["mse"]


def test_sequence_aligned_sidecar_cli_writes_json_and_markdown(tmp_path) -> None:
    output_path = tmp_path / "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar.json"
    markdown_path = tmp_path / "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar.md"

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
            "--remap-capacity",
            "10",
            "--interface-strength",
            "2.5",
            "--remap-recovery",
            "0.7",
            "--sidecar-hash-dim",
            "16",
            "--sidecar-ngram-max",
            "2",
            "--sidecar-scale",
            "0.35",
            "--alignment-hash-dim",
            "16",
            "--alignment-scale",
            "0.30",
            "--alignment-profile-scale",
            "0.20",
            "--alignment-token-scale",
            "0.75",
        ]
    )

    saved = json.loads(output_path.read_text())
    assert saved == payload
    markdown = markdown_path.read_text()
    assert "Toy Quotient + GPA Sparse Dictionary Sequence-Aligned Sidecar" in markdown
    assert "quotient_gpa_sparse_dictionary_sequence_aligned_sidecar_remap" in markdown
    assert "DWA-KD" in markdown
