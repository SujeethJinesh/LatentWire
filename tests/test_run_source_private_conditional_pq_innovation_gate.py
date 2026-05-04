from __future__ import annotations

import json

import numpy as np

from scripts.run_source_private_conditional_pq_innovation_gate import BASIS_VIEWS, CONDITIONING_MODES, _derangement, run_gate


def test_derangement_moves_every_candidate() -> None:
    perm = _derangement(4, seed=11)

    assert sorted(int(value) for value in perm) == [0, 1, 2, 3]
    assert np.all(perm != np.arange(4))


def test_cli_exposes_less_diagnostic_basis_views() -> None:
    assert "semantic" in BASIS_VIEWS
    assert "no_diag" in BASIS_VIEWS
    assert "public_zscore" in CONDITIONING_MODES
    assert "public_svd_whiten" in CONDITIONING_MODES


def test_conditional_pq_innovation_gate_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "conditional_pq",
        train_examples=48,
        eval_examples=24,
        train_seed=5,
        eval_seed=6,
        train_start_index=1000,
        eval_start_index=0,
        train_family_set="all",
        eval_family_set="all",
        diagnostic_table_mode="legacy",
        candidates=4,
        feature_dim=64,
        anchor_count=16,
        basis_view="shared_text",
        source_topk=16,
        target_topk=8,
        budget_bytes=2,
        variant="protected_hadamard",
        remap_slot_seed=101,
        ridge=1e-2,
        fit_intercept=False,
        mask_repeats=1,
        codebook_iterations=4,
        seed=5,
        bootstrap_samples=64,
    )

    summary = json.loads((tmp_path / "conditional_pq" / "summary.json").read_text())
    assert payload["gate"] == "source_private_conditional_pq_innovation"
    assert summary["summary"]["train_eval_id_intersection_count"] == 0
    assert summary["summary"]["n"] == 24
    assert "deranged_public_basis" in summary["summary"]["conditions"]
    assert "opaque_slot_basis" in summary["summary"]["conditions"]
    assert "public_condition_only" in summary["summary"]["conditions"]
    assert "same_answer_slot_wrong_row_source" in summary["summary"]["conditions"]
    assert (tmp_path / "conditional_pq" / "summary.md").exists()
    assert (tmp_path / "conditional_pq" / "predictions.jsonl").exists()


def test_public_zscore_conditioning_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "conditional_pq_public_zscore",
        train_examples=48,
        eval_examples=24,
        train_seed=7,
        eval_seed=8,
        train_start_index=1000,
        eval_start_index=0,
        train_family_set="core",
        eval_family_set="holdout",
        diagnostic_table_mode="plausible_decoys",
        candidates=4,
        feature_dim=64,
        anchor_count=16,
        basis_view="semantic",
        source_topk=16,
        target_topk=8,
        budget_bytes=2,
        variant="protected_hadamard",
        remap_slot_seed=101,
        ridge=1e-2,
        fit_intercept=False,
        mask_repeats=1,
        codebook_iterations=4,
        seed=7,
        bootstrap_samples=64,
        conditioning_mode="public_zscore",
    )

    summary = json.loads((tmp_path / "conditional_pq_public_zscore" / "summary.json").read_text())
    assert payload["conditioning_mode"] == "public_zscore"
    assert summary["summary"]["train_eval_id_intersection_count"] == 0
    assert summary["conditioning_mode"] == "public_zscore"
    assert (tmp_path / "conditional_pq_public_zscore" / "predictions.jsonl").exists()


def test_public_svd_whiten_conditioning_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "conditional_pq_public_svd_whiten",
        train_examples=48,
        eval_examples=24,
        train_seed=7,
        eval_seed=8,
        train_start_index=1000,
        eval_start_index=0,
        train_family_set="core",
        eval_family_set="holdout",
        diagnostic_table_mode="plausible_decoys",
        candidates=4,
        feature_dim=64,
        anchor_count=16,
        basis_view="semantic",
        source_topk=16,
        target_topk=8,
        budget_bytes=2,
        variant="protected_hadamard",
        remap_slot_seed=101,
        ridge=1e-2,
        fit_intercept=False,
        mask_repeats=1,
        codebook_iterations=4,
        seed=7,
        bootstrap_samples=64,
        conditioning_mode="public_svd_whiten",
    )

    summary = json.loads((tmp_path / "conditional_pq_public_svd_whiten" / "summary.json").read_text())
    assert payload["conditioning_mode"] == "public_svd_whiten"
    assert summary["summary"]["train_eval_id_intersection_count"] == 0
    assert summary["conditioning_mode"] == "public_svd_whiten"
    assert (tmp_path / "conditional_pq_public_svd_whiten" / "predictions.jsonl").exists()
