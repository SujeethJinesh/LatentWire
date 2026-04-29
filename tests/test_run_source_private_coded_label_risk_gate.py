from __future__ import annotations

import json

from scripts import run_source_private_coded_label_risk_gate as gate


def test_transform_hashes_change_expected_surfaces() -> None:
    payload = gate.run_gate(
        examples=48,
        candidates=4,
        family_set="all",
        seeds=[29],
        budget_bytes=2,
    )

    assert payload["pass_gate"] is True
    rows = {row["transform"]: row for row in payload["rows"]}

    assert rows["baseline"]["hash_gate"] is True
    assert rows["label_rename"]["hash_change_checks"]["exact_id_same_as_baseline"] is True
    assert rows["label_rename"]["hash_change_checks"]["label_changed_vs_baseline"] is True
    assert rows["label_rename"]["hash_change_checks"]["candidate_order_changed_vs_baseline"] is False
    assert rows["diagnostic_code_remap"]["hash_change_checks"]["codebook_changed_vs_baseline"] is True
    assert rows["candidate_pool_permutation"]["hash_change_checks"]["label_changed_vs_baseline"] is True
    assert rows["candidate_pool_permutation"]["hash_change_checks"]["candidate_order_changed_vs_baseline"] is True
    assert rows["label_code_order_composed"]["hash_change_checks"]["label_changed_vs_baseline"] is True
    assert rows["label_code_order_composed"]["hash_change_checks"]["candidate_order_changed_vs_baseline"] is True
    assert rows["label_code_order_composed"]["hash_change_checks"]["codebook_changed_vs_baseline"] is True


def test_coded_label_risk_gate_controls_stay_at_target() -> None:
    payload = gate.run_gate(
        examples=48,
        candidates=4,
        family_set="all",
        seeds=[29, 31],
        budget_bytes=2,
    )

    assert payload["pass_gate"] is True
    for row in payload["rows"]:
        summary = row["summary"]
        target = summary["target_accuracy"]
        assert summary["matched_accuracy"] == 1.0
        assert summary["best_source_destroying_control_accuracy"] <= target + 0.03
        assert summary["best_reviewer_negative_control_accuracy"] <= target + 0.03
        assert summary["min_positive_oracle_accuracy"] >= summary["matched_accuracy"]


def test_markdown_and_cli_payload_are_serializable(tmp_path) -> None:
    payload = gate.run_gate(
        examples=16,
        candidates=4,
        family_set="core",
        seeds=[29],
        budget_bytes=2,
    )
    prediction_rows = payload.pop("prediction_rows")
    output = tmp_path / "summary.md"

    gate._write_markdown(output, payload)
    encoded = json.dumps(payload, sort_keys=True)

    assert "Source-Private Coded-Label Risk Gate" in output.read_text(encoding="utf-8")
    assert '"pass_gate": true' in encoded
    assert prediction_rows
