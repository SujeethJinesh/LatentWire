from __future__ import annotations

import numpy as np

from scripts.run_source_private_hidden_repair_packet_smoke import make_benchmark
from scripts.run_source_private_pq_control_regularized_receiver import (
    _derangement,
    run_gate,
)


def test_derangement_moves_every_candidate() -> None:
    perm = _derangement(4, seed=7)

    assert sorted(int(value) for value in perm) == [0, 1, 2, 3]
    assert np.all(perm != np.arange(4))


def test_pq_control_regularized_receiver_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "pq_receiver",
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
        budget_bytes=2,
        variant="protected_hadamard",
        remap_slot_seed=101,
        ridge=1e-2,
        receiver_ridge=1e-2,
        candidate_view="slot",
        fit_intercept=False,
        opq_iterations=1,
        seed=5,
        matched_weight=4.0,
        control_weight=2.0,
        target_weight=2.0,
        deranged_weight=0.5,
        random_rounds=1,
        bootstrap_samples=64,
        tolerance_vs_l2=0.20,
        conditions=None,
    )

    assert payload["gate"] == "source_private_pq_control_regularized_receiver"
    assert payload["train_eval_id_intersection_count"] == 0
    assert payload["summary"]["n"] == 24
    assert "deranged_public_table" in payload["summary"]["conditions"]
    assert (tmp_path / "pq_receiver" / "summary.md").exists()
    assert (tmp_path / "pq_receiver" / "predictions.jsonl").exists()


def test_benchmark_ids_can_be_disjoint_by_start_index() -> None:
    train = make_benchmark(examples=4, candidates=4, seed=1, family_set="all", start_index=100)
    eval_rows = make_benchmark(examples=4, candidates=4, seed=2, family_set="all", start_index=0)

    assert not {row.example_id for row in train}.intersection(row.example_id for row in eval_rows)
