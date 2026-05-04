from __future__ import annotations

import json

from scripts.build_source_private_conditional_pq_integrity_threshold_gate import run_gate


def test_conditional_pq_integrity_threshold_gate_writes_artifacts(tmp_path) -> None:
    payload = run_gate(
        output_dir=tmp_path / "conditional_pq_integrity",
        train_examples=32,
        integrity_select_examples=8,
        eval_examples=12,
        train_seed=7,
        eval_seed=8,
        train_start_index=1000,
        eval_start_index=0,
        train_family_set="core",
        eval_family_set="holdout",
        diagnostic_table_mode="plausible_decoys",
        candidates=4,
        feature_dim=64,
        anchor_count=12,
        basis_view="semantic",
        source_topk=12,
        target_topk=8,
        conditioning_mode="public_zscore",
        budget_bytes=2,
        variant="protected_hadamard",
        remap_slot_seed=101,
        encoder_ridge=1e-2,
        receiver_ridge=1e-2,
        receiver_noop_weight=0.01,
        fit_intercept=False,
        mask_repeats=1,
        codebook_iterations=3,
        seed=7,
        bootstrap_samples=32,
    )

    summary = json.loads((tmp_path / "conditional_pq_integrity" / "summary.json").read_text())
    predictions = (tmp_path / "conditional_pq_integrity" / "predictions.jsonl").read_text().splitlines()
    threshold_rows = (tmp_path / "conditional_pq_integrity" / "threshold_rows.csv").read_text().splitlines()

    assert payload["gate"] == "source_private_conditional_pq_integrity_threshold_gate"
    assert summary["headline"]["eval_rows"] == 12
    assert summary["headline"]["select_rows"] == 8
    assert summary["selected_integrity_rule"]["score_name"]
    assert summary["systems_accounting"]["payload_bytes"] == 2
    assert summary["systems_accounting"]["framed_packet_bytes_estimate"] == 5
    assert "source" in summary["summary"]["metrics"]
    assert len(predictions) == 12 * len(summary["summary"]["conditions"])
    assert len(threshold_rows) > 1
    assert (tmp_path / "conditional_pq_integrity" / "summary.md").exists()
    assert (tmp_path / "conditional_pq_integrity" / "manifest.json").exists()
