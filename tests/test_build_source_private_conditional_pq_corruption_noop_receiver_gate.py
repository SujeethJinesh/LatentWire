from __future__ import annotations

import json

from scripts.build_source_private_conditional_pq_corruption_noop_receiver_gate import CONDITIONS, run_gate


def test_conditional_pq_corruption_noop_receiver_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "conditional_pq_corruption_noop",
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
        conditioning_mode="public_zscore",
        budget_bytes=2,
        variant="protected_hadamard",
        remap_slot_seed=101,
        encoder_ridge=1e-2,
        receiver_ridge=1e-2,
        receiver_noop_weight=0.1,
        fit_intercept=False,
        mask_repeats=1,
        codebook_iterations=4,
        seed=7,
        bootstrap_samples=64,
    )

    summary = json.loads((tmp_path / "conditional_pq_corruption_noop" / "summary.json").read_text())
    predictions = (tmp_path / "conditional_pq_corruption_noop" / "predictions.jsonl").read_text().splitlines()

    assert payload["gate"] == "source_private_conditional_pq_corruption_noop_receiver"
    assert summary["conditioning_mode"] == "public_zscore"
    assert summary["receiver"]["noop_weight"] == 0.1
    assert "packet_candidate_similarity" in summary["receiver"]["feature_names"]
    assert "source" in summary["receiver"]["training_diagnostics"]
    assert "target_only" in summary["receiver"]["training_diagnostics"]
    assert summary["systems_accounting"]["payload_bytes"] == 2
    assert summary["systems_accounting"]["framed_packet_bytes_estimate"] == 5
    assert summary["summary"]["train_eval_id_intersection_count"] == 0
    assert summary["summary"]["exact_id_parity"]
    assert "candidate_roll" in summary["summary"]["conditions"]
    assert "same_answer_slot_wrong_row_source" in summary["summary"]["conditions"]
    assert len(predictions) == 24 * len(CONDITIONS)
    assert (tmp_path / "conditional_pq_corruption_noop" / "summary.md").exists()
    assert (tmp_path / "conditional_pq_corruption_noop" / "manifest.json").exists()
