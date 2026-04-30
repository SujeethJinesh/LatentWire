from __future__ import annotations

import json

import numpy as np

from scripts.run_source_private_conditional_pq_innovation_gate import _derangement, run_gate


def test_derangement_moves_every_candidate() -> None:
    perm = _derangement(4, seed=11)

    assert sorted(int(value) for value in perm) == [0, 1, 2, 3]
    assert np.all(perm != np.arange(4))


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
    assert (tmp_path / "conditional_pq" / "summary.md").exists()
    assert (tmp_path / "conditional_pq" / "predictions.jsonl").exists()
